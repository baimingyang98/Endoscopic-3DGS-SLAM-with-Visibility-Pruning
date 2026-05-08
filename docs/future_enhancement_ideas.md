# Future Enhancement Ideas — Detailed Specification

This document expands each enhancement idea into enough detail that you can defend it under questioning, estimate the work, and identify what could go wrong.

Each idea is rated on:
- **Novelty** (1-5): how unique vs. published 2025-2026 work
- **Feasibility** (1-5): how realistic in 6 months given your resources
- **Publication value** (1-5): venue this could target
- **Risk** (1-5): chance of negative result

---

## Idea 1 — Active Perception via Coverage Confidence Map ★ STRONGEST

**Novelty: 5/5 | Feasibility: 4/5 | Publication value: 5/5 | Risk: 2/5**

### The gap
All published 3DGS-SLAM methods (EndoGSLAM, EndoFlow-SLAM, Endo-2DTAM, NRGS-SLAM, ColonSplat) treat reconstruction as a passive task. The output is a Gaussian map and trajectory; the system does not communicate **uncertainty or coverage adequacy** to the surgeon. In colonoscopy specifically, missed regions correlate with missed polyps and are a recognized clinical metric (cecal intubation completeness, withdrawal time analysis).

### The core idea
Use the visibility statistics already extracted from our modified rasterizer to compute a **per-pixel reliability score** for the live rendered view. This reliability map is overlaid in real time on the endoscope display.

### Math
For each pixel p in the current rendered frame:

C(p) = sum over Gaussians i contributing to p of (alpha_i · T_p · S_i)

where S_i is a per-Gaussian "stability" score:

S_i = (mean(V_i over W frames)) · (1 - normalized_variance(V_i)) · clip(age_i / 20, 0, 1)

This combines (1) how often the Gaussian was visible, (2) how consistently it contributed (low variance = stable), (3) Gaussian maturity (newly added Gaussians are less trusted).

### Visualization
Three colors:
- Green (C > τ_high): "well reconstructed, trust this geometry"
- Yellow (τ_low < C < τ_high): "marginal, more views helpful"
- Red (C < τ_low): "needs revisit"

### Why it's defensible
1. **Uses signals already computed** — minimal compute overhead
2. **Mathematically principled** — based on rendering equation, not ad-hoc
3. **Clinically meaningful** — connects to coverage / withdrawal-time literature
4. **Evaluable** — can compute precision/recall against ground-truth coverage on C3VDv2 (which has full mesh ground truth)

### What you would write in the paper
- Section 3.4: "Visibility-Derived Coverage Confidence"
- Section 4.3: "Coverage Completeness Evaluation" — table comparing coverage % vs ground-truth mesh on C3VD scenes
- Section 4.4: "Surgeon Study" (optional, ambitious): show 2-3 surgeons rendered overlays vs baseline, ask which gives more useful guidance

### Risks
- (Risk 1) The confidence map might just track depth — need to show it captures something extra
- (Risk 2) Surgeon study is hard to organize; may need to settle for proxy metric (correlation with ground-truth coverage)

### Deliverable timeline
- Week 1-2: implement C(p) computation in render pipeline (~200 lines)
- Week 3: validate against C3VDv2 ground-truth meshes
- Week 4-5: write the section + figures

---

## Idea 2 — Learned Lifecycle Classifier ★ STRONG

**Novelty: 4/5 | Feasibility: 5/5 | Publication value: 4/5 | Risk: 2/5**

### The gap
Every existing pruning method uses hand-tuned thresholds. LightGaussian (NeurIPS 2024) uses heuristic global significance. Prune Wisely (Feb 2026) uses reconstruction-aware heuristics. None has trained a learned classifier driven by per-Gaussian dynamics.

### The core idea
Replace the hand-tuned dual-mask pruning rule with a small MLP that predicts an action for each Gaussian per mapping iteration:
- **keep** (do nothing)
- **degenerate** (multiply opacity by η, learned)
- **deforming** (set deform_mask, eligible for δxyz)

### Architecture
```
Input features (per Gaussian, all already computed):
- mean(V_i over W=15 frames)            (scalar)
- variance(V_i over W=15 frames)        (scalar)
- current opacity (sigmoid)             (scalar)
- log scale                              (scalar)
- age (frames since birth, normalized)   (scalar)
- distance_to_observed_surface           (scalar)
- local_color_gradient_contribution      (scalar)

7-dim input -> MLP(64, 32, 3) -> softmax over {keep, degenerate, deform}
```

Tiny MLP (~1k params), runs on every Gaussian per iteration in batched fashion.

### Training strategy
**Self-supervised with future-lookahead labels:**
- During an offline training pass, simulate "what if I removed this Gaussian at frame t?" by zeroing its opacity for the next 10 frames
- If PSNR/RMSE improves over baseline → label "degenerate"
- If significantly worsens → label "keep"
- Otherwise → middle case

This is essentially counterfactual supervision; doesn't need human labels.

### Why it's defensible
1. **Replaces 5 hand-tuned hyperparameters with one learned model**
2. **Generalizes across datasets** — train on C3VD train split, evaluate zero-shot on C3VD test split + StereoMIS
3. **Marginal compute overhead** — 1k-param MLP is negligible compared to rendering
4. **Interpretable** — feature importance analysis reveals what makes a Gaussian "good" or "bad"

### Risks
- (Risk 1) Lookahead labeling may be noisy → need thoughtful loss weighting
- (Risk 2) Could overfit to one dataset → need StereoMIS validation
- (Risk 3) Improvement might be marginal vs hand-tuned thresholds → need to show generalization advantage

### Deliverable timeline
- Week 1: data collection script (Gaussian features + lookahead labels)
- Week 2: MLP training, hyperparameter sweep
- Week 3: integrate into SLAM pipeline, replace hand-tuned pruning
- Week 4: cross-dataset validation

---

## Idea 3 — Joint Visibility-Deformation Coupling ★ MODERATE

**Novelty: 3/5 | Feasibility: 4/5 | Publication value: 4/5 | Risk: 3/5**

### The gap
NRGS-SLAM (Feb 2026) uses Bayesian deformation probability — but treats deformation independently from visibility statistics. Our approach uses visibility variance to detect deformation, but currently treats Gaussians independently. There's a coupling story to be told: a Gaussian with high visibility variance is *probably* deforming **if its neighborhood deforms similarly**, otherwise it's probably a floater.

### The core idea
Disambiguate "high visibility variance" using spatial neighborhood:
- Compute kNN graph among Gaussians
- For each Gaussian, compute *neighbor deformation correlation* — do my neighbors also have high variance and similar δxyz direction?
- If yes → real tissue deformation → keep and apply δxyz
- If no → likely floater → degenerate

### Math
For Gaussian i with k nearest neighbors N(i):

D_corr(i) = (1/k) sum over j in N(i) of cosine_similarity(δxyz_i, δxyz_j) · sign(var(V_j) > τ_var)

Then:
- If var(V_i) > τ_var AND D_corr(i) > τ_corr → "deforming"
- If var(V_i) > τ_var AND D_corr(i) < τ_corr → "floater"
- Otherwise → "static"

### Connection to ARAP regularization
The "Multi-Level Geometry Regularization" paper (Feb 2026) uses As-Rigid-As-Possible regularization. We could combine: ARAP enforces local rigidity for static regions, while our visibility-correlation detects which neighborhoods are exempt from ARAP (because they're truly deforming).

### Why it's defensible
1. **Unifies two scattered ideas** (visibility variance + spatial deformation field) into one coherent framework
2. **Resolves the disambiguation problem** that plagues all isolated-Gaussian deformation methods
3. **Connects to established graphics techniques** (ARAP, Laplacian regularization)

### Risks
- (Risk 1) kNN graph computation per frame is expensive — may need spatial hashing
- (Risk 2) Hard to evaluate isolated effect (entangled with deformation modeling)

### Deliverable timeline
- Week 1: implement spatial kNN with frame-coherent caching
- Week 2: implement correlation-based gating
- Week 3: ablation on StereoMIS (with vs without correlation)

---

## Idea 4 — Task-Aware Reconstruction Loss ★ MODERATE-LOW

**Novelty: 3/5 | Feasibility: 5/5 | Publication value: 3/5 | Risk: 3/5**

### The gap
All current loss functions (L1 + SSIM + depth + flow) target photometric/geometric quality. None optimizes for downstream surgical tasks.

### The core idea
Augment the rendering loss with a task-feature consistency term using a frozen pretrained network:

L_total = L_photo + λ_task · L_task

L_task = || f(rendered_image) - f(ground_truth_image) ||²

where f is one of:
- **Surgical tool segmentation** model (Endovis17/18 pretrained)
- **Polyp localization** model (PICCOLO/CVC-ClinicDB pretrained)
- **DINO/SAM features** (general-purpose pretrained)

### Why it works (hypothesis)
Gaussians that contribute to task-relevant image features (tool boundaries, polyp edges) get prioritized; Gaussians representing photometrically-correct-but-task-irrelevant detail (specular highlights, motion blur) get deprioritized.

### Why it's defensible
1. **Quick to implement** (~50 lines, plug-and-play)
2. **Connects 3DGS-SLAM to active research in surgical scene understanding**
3. **Multiple downstream tasks** can be tested → strong ablation story

### Risks
- (Risk 1) Could be considered "just adding another loss" — need a strong narrative
- (Risk 2) The pretrained network may be sensitive to rendering artifacts unrelated to tasks
- (Risk 3) Improvement may be marginal on standard PSNR/RMSE metrics → must include task-level evaluation

### Deliverable timeline
- Week 1: integrate frozen Endovis-pretrained tool seg model
- Week 2: ablation on C3VD (no tools) vs StereoMIS (with tools)
- Week 3: add downstream task evaluation (tool seg accuracy on rendered novel views)

---

## Idea 5 — Multi-View Consistency Filter ★ BACKUP

**Novelty: 2/5 | Feasibility: 5/5 | Publication value: 2/5 | Risk: 1/5**

### The gap
Photometric inconsistency from non-Lambertian endoscopic surfaces (specular highlights from wet tissue) creates view-dependent floaters. EndoFlow-SLAM addresses this with depth regularization but not with multi-view photometric checks.

### The core idea
For each Gaussian, compute the variance of its rendered color across recent N keyframes. High color variance from the same Gaussian = view-dependent artifact = candidate for removal.

This is a small extension of Innovation 1's visibility tracking: instead of (or in addition to) tracking V_i, also track the rendered color contribution C_i across frames.

### Why it's defensible
- Specular handling is a known endoscopic problem
- Multi-view photometric consistency is well-established in classical MVS
- Easy to integrate with existing visibility buffer

### Why it's a backup
- Less novel than ideas 1-4
- Modest publication value
- But almost zero risk: it will work, just may not be exciting

### Use case
Add as a minor contribution to bolster the paper if main ideas don't pan out.

---

## Recommended Combination for Q1 Paper

**Single coherent paper structured around three reinforcing ideas:**

1. **Foundation (already done):** CUDA-level visibility extraction
2. **Method (Idea 2):** Learned lifecycle classifier replaces hand-tuned thresholds
3. **Application (Idea 1):** Coverage confidence map for surgical guidance

This gives you:
- One technical core (lifecycle management)
- One unique application angle (active perception)
- One foundational tool (rasterizer modification)

Ideas 3 and 4 become Section 4.3 / 4.4 ablations or future work.

This combination targets:
- **MedIA / IEEE TMI** (clinical relevance from coverage map)
- **MICCAI 2026** main conference (workshop track if rejected)
- Backup: **WACV / CVPR Workshop on Surgical Computer Vision**

---

## Questions to Anticipate from Tutor

| Question | Suggested answer |
|---------|------------------|
| "How is this different from NRGS-SLAM?" | "NRGS-SLAM treats deformation as an isolated per-Gaussian property. We use visibility statistics as a unified signal that drives pruning, deformation, AND surgical guidance — three uses from one signal." |
| "How is the coverage map different from depth uncertainty?" | "Depth uncertainty is per-pixel and instantaneous. Our coverage map is per-Gaussian and accumulates over the entire SLAM trajectory — it captures whether a region has been observed enough times from enough angles." |
| "What's the clinical evidence this matters?" | "Withdrawal time and cecal intubation rate are recognized colonoscopy quality metrics. Studies show >25% of polyps are missed due to incomplete inspection. We're providing a real-time signal correlated with this." |
| "Why not just use 2DGS like Endo-2DTAM?" | "2DGS improves geometric fidelity but doesn't help with map management or surgical guidance. The two are orthogonal — our framework would also benefit from a 2DGS substrate, but the contribution is at a different layer." |
| "Why visibility from CUDA, not from a separate MLP like NVGS?" | "NVGS predicts visibility for occlusion culling at render time. We extract actual rendered contribution as a side product of normal rendering — zero overhead, zero extra parameters, no training needed for the extraction itself." |
