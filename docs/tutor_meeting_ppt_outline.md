# Tutor Meeting PPT — Slide-by-Slide Outline

**Goal of this meeting:** convince the tutor that your work has a clear publication path and is worth her supervision time.

**Strategic message:** "I have a working system, I know the competitive landscape, and I have a defensible publishable angle that no current method addresses."

**Suggested length:** 12-15 slides, ~20 minutes presentation + 10 minutes Q&A.

---

## Slide 1 — Title

**Visibility-Driven Gaussian Map Management for Endoscopic SLAM**

Subtitle: *From floater removal to active perception for surgical guidance*

Your name | Date | Tutor name

---

## Slide 2 — Problem and motivation (1 minute)

**The clinical problem**
- During endoscopy, surgeons need real-time 3D understanding for navigation and inspection coverage
- Current 3DGS-SLAM systems (EndoGSLAM, EndoFlow-SLAM) optimize photometric quality but produce no clinical reliability signal
- They cannot tell the surgeon: "this region is well-reconstructed, that region needs another pass"

**The technical problem**
- Gaussian maps accumulate "floater" primitives that hurt depth accuracy and tracking
- Existing pruning is opacity-only and heuristic — no principled signal of which Gaussians are reliable

**One-line thesis (write this on the slide)**
> "We use per-Gaussian visibility statistics — a free byproduct of rendering — as a unifying signal for map quality, lifecycle management, and surgical guidance."

---

## Slide 3 — Competitive Landscape Snapshot (2 minutes)

A 2-column table on the slide:

| Year | Method | Key idea | Limitation |
|------|--------|----------|-----------|
| 2024 | EndoGSLAM | Simplified 3DGS for endoscopy | No deformation, no BA, basic pruning |
| 2025 | Endo-2DTAM | 2D Gaussians + surface normals | Pure quality focus, not surgical context |
| 2025 | EndoFlow-SLAM | Optical flow + depth regularization | No quality/uncertainty output |
| 2025 | PR-ENDO | Physically-based relighting | Reconstruction only, no SLAM control |
| 2026 | NRGS-SLAM | Bayesian deformation probability | Expensive dual-rendering, no surgical use |
| 2026 | ColonSplat | Peristaltic motion modeling | Synthetic data, no clinical signal |

**Key message:** *"The field is converging on photometric/geometric quality. No one is using SLAM as a surgical decision-support signal."*

---

## Slide 4 — System overview (1 minute)

[Insert workflow diagram from `docs/workflow_diagram.md`]

Emphasize three things:
1. Standard EndoGSLAM backbone (proves I implemented it correctly)
2. CUDA-level rasterizer modification (one atomicAdd, exposes visibility V_i)
3. Three innovation modules connected by visibility statistics

---

## Slide 5 — Technical contribution 1: per-Gaussian visibility from CUDA (2 minutes)

**The signal**
For Gaussian i and pixel p, blending weight is α_i · T_p (alpha · transmittance). This is already computed for color accumulation.

**The modification**
Add one line in the CUDA kernel:
```cuda
atomicAdd(&out_gaussian_visibility[gauss_id], alpha * T);
```

**Why this matters**
- Zero extra compute (single atomicAdd per pixel-Gaussian pair, free piggyback on existing alpha-blending)
- Continuous score (not binary radius>0)
- Captures actual rendering contribution, not just frustum membership
- Makes downstream decisions principled instead of heuristic

**Slide visual:** rasterizer pipeline schematic with the added line highlighted.

---

## Slide 6 — Technical contribution 2: dual-mask pruning + lifecycle (3 minutes)

**Two complementary signals**
1. *Visibility floater*: mean V_i over recent W=15 frames is below threshold
2. *Distance floater*: Gaussian projects in front of observed depth surface (GS-SLAM Eq. 9)

**Combined mask** triggers gradual opacity degeneration: σ_i ← σ_i · η (η=0.90)
- Soft delete, optimizer can recover
- Standard opacity threshold then sweeps faded Gaussians

**Three-way classifier** (using visibility variance)
- STATIC: high mean V, low variance → standard treatment
- DEFORMING: high mean V, high variance → eligible for position offset (Innovation 3)
- FLOATER: low mean V → opacity degeneration

**Slide visual:** scatter plot or schematic showing the three-way clustering in (mean, variance) space.

---

## Slide 7 — Technical contribution 3: periodic BA + deformation (1 minute, brief)

**Periodic BA** — every 50 frames, joint optimization of 5 keyframes for 20 iterations. ~1.5% overhead.

**Deformation modeling** — per-Gaussian δxyz offset, applied only to DEFORMING-class Gaussians. Regularized for temporal smoothness.

Brief because these are smaller contributions — the visibility signal is the headline.

---

## Slide 8 — Current results (2 minutes)

**Setup:** sigmoid_t3_a sequence, 537 frames, C3VD dataset

| Configuration | PSNR ↑ | LPIPS ↓ | RMSE ↓ | ATE ↓ |
|---|---|---|---|---|
| Baseline EndoGSLAM | 22.137 | **0.288** | 1.814 | 0.375 |
| + BA only | 22.161 | 0.291 | 1.805 | 0.375 |
| + BA + Prune (η=0.80) | 22.143 | 0.369 | 1.883 | **0.354** |
| + BA + Prune (η=0.90) | **22.461** | 0.339 | **1.617** | 0.463 |
| + BA + Prune (η=0.95) | 22.358 | 0.314 | 1.779 | 0.453 |

**Key findings to verbally state**
- η=0.90 gives **+1.5% PSNR and -10.9% RMSE** vs baseline — best reconstruction
- η=0.80 gives **best tracking (-5.6% ATE)** but hurts perceptual quality
- η is a clean Pareto knob between tracking accuracy and rendering quality
- BA alone is neutral (validates pose drift was already minor on this scene)

**Status disclosure (be honest):**
- 10-scene C3VD ablation in progress
- StereoMIS preprocessing pipeline being built (Innovation 3 will be properly evaluated there)

---

## Slide 9 — Honest assessment of competitive position (2 minutes)

**Where we currently stand**
- Innovation 1 (visibility-driven pruning): unique to endoscopy SLAM, not directly published
- Innovation 2 (periodic BA): similar to GS-SLAM's BA, mild novelty
- Innovation 3 (deformation): NRGS-SLAM (Feb 2026) does this with stronger formulation

**What the current paper is not**
- Not a state-of-the-art chase: NRGS-SLAM and Endo-2DTAM beat us on raw metrics
- Not pure photometric improvement: EndoFlow-SLAM is closer to that

**What we could uniquely contribute** (transition slide to future ideas)
- A **clinical-relevance angle** that no Gaussian-SLAM paper currently offers
- A **principled lifecycle framework** unifying pruning, deformation handling, and quality estimation
- A **learned (not heuristic) replacement** for hand-tuned thresholds

---

## Slide 10 — Future direction 1: Active perception for surgical coverage (3 minutes)

**The gap (no published method addresses this)**
All current 3DGS-SLAM produces a static reconstruction. None tells the surgeon *where to look next*.

**The proposal**
Use per-Gaussian visibility statistics to compute a **per-pixel coverage confidence map** in real time:
- Pixel covered by Gaussians with high mean V over many frames → green (trustworthy)
- Pixel covered only by low-V or recently-added Gaussians → red (revisit needed)

**Concrete output**
Overlay rendered onto the live endoscope view showing which surface patches need another pass for reliable reconstruction. Connects directly to colonoscopy *withdrawal time / coverage completeness* — a recognized clinical metric.

**Why this is a fresh contribution**
- No 3DGS-SLAM paper does this
- Connects to clinical literature on cecal intubation completeness, polyp detection rates
- Easily evaluable: simulated polyp localization in C3VDv2 with vs without our coverage guidance

**Publication target:** MedIA, IEEE TMI (clinical relevance opens journal venues)

---

## Slide 11 — Future direction 2: Learned lifecycle management (2 minutes)

**The gap**
All pruning thresholds (η, γ, vis_threshold, var_threshold) are hand-tuned per dataset. No method has learned them.

**The proposal**
Train a small MLP that takes per-Gaussian features → predicts "kill | keep | promote-to-deforming":

Inputs (cheap, all already computed):
- Mean and variance of recent visibility V_i
- Opacity, log-scale, age (frames since birth)
- Distance to nearest depth surface
- Local color-gradient contribution at pixels where V_i > 0

Training:
- Self-supervised: ground-truth labels from "did removing this Gaussian later hurt PSNR/RMSE?"
- Or contrastive: encourage label stability across frames

**Why it works**
- Replaces ~5 hand-tuned thresholds with one learned model
- Generalizes across datasets (one MLP, no per-dataset tuning)
- Connects to the broader trend of learned compression/pruning (LightGaussian, Prune Wisely 2026)

---

## Slide 12 — Future direction 3: Joint visibility-deformation modeling (2 minutes)

**The gap**
NRGS-SLAM uses Bayesian deformation probability but doesn't tie it to visibility statistics. There's a strong coupling we can exploit.

**The proposal**
A unified Gaussian state with two correlated signals:
- **Visibility variance** = "I am observed inconsistently" (could be deforming OR could be a floater)
- **Deformation magnitude** = "I am moving in 3D"

Disambiguate by checking if the *neighborhood* deforms similarly:
- High variance + locally consistent neighborhood deformation → real deformation
- High variance + isolated → floater

**Implementation**
- Spatial smoothness regularizer for δxyz across kNN Gaussians (existing tech)
- Anisotropic regularization based on local visibility patterns (novel)

**Connects Innovation 1 and Innovation 3 into a single coherent contribution**

---

## Slide 13 — Future direction 4: Task-aware reconstruction loss (1 minute)

**The gap**
Every paper minimizes photometric L1+SSIM and depth L1. None optimizes for downstream surgical tasks.

**The proposal**
Replace/augment the rendering loss with a task-aware term:
- L_task = || f_seg(rendered) - f_seg(ground_truth) ||
- where f_seg is a frozen surgical-tool / polyp / vessel segmentation network

**Hypothesis:** Gaussians that contribute to task-relevant features will be preserved; cosmetic floaters (e.g., specular highlights) will be down-weighted.

**Quick win:** uses pretrained networks, no new training required.

---

## Slide 14 — Proposed contribution narrative for the paper (2 minutes)

**Single tight thesis (replaces "three loose innovations")**

> *"Visibility-Aware Lifecycle Management of Gaussians for Endoscopic SLAM:*
> *from passive reconstruction quality to active surgical coverage guidance"*

**Three layers**
1. **Foundation:** CUDA-level visibility extraction (free)
2. **Method:** Visibility-driven dual-mask pruning + learned lifecycle classifier
3. **Application:** Real-time coverage confidence map for surgical guidance

**Why this narrative is publishable**
- Single technical thread (not three loose innovations)
- One unique insight (visibility as a lifecycle signal)
- Real clinical application (coverage guidance)
- Multi-layer evaluation: photometric (PSNR), geometric (RMSE), pose (ATE), task-level (coverage completeness)

---

## Slide 15 — Roadmap and ask (2 minutes)

**Months 1-2 (in progress)**
- 10-scene C3VD ablation (50% complete)
- StereoMIS preprocessing + Innovation 3 evaluation
- Baseline reproduction (EndoFlow-SLAM, Endo-2DTAM)

**Months 3-4**
- Implement learned lifecycle MLP (Future direction 2)
- Add coverage confidence map module (Future direction 1)
- Begin task-aware loss experiments (Future direction 4)

**Months 5-6**
- Multi-dataset benchmark (C3VD + StereoMIS + Hamlyn)
- Statistical significance testing
- Paper writing

**Specific asks to the tutor**
1. Feedback on which of the four future directions has highest publication value (rank them)
2. Connection to clinical collaborators if pursuing the surgical-coverage direction
3. Access to GPU resources beyond Colab Enterprise + RTX 5060 (multi-day runs needed)
4. Suggestion of target venue: workshop submission first (MICCAI workshops, deadline ~April) vs direct journal (MedIA/TMI)

**Closing line**
> "I have a working system, I understand the landscape, and I see a path to a publication that's not just incremental. I want guidance to choose the strongest direction and execute it well."

---

## Speaker tips

1. **Slide 3 (landscape) and slide 9 (honest assessment) are critical.** They show maturity. Tutors are tired of students who don't know the literature. Owning the gaps and showing you've identified them earns trust.

2. **Don't oversell current results.** PSNR +1.5% is small. Frame the current results as "validating that the visibility signal works"; the value comes from what you do with it next (slides 10-13).

3. **Lead with the future ideas if asked.** If she asks "what are you proposing for novelty?", jump straight to slides 10-12 — the active perception angle is your strongest card.

4. **Be ready to drop Innovation 3.** If the tutor says "NRGS-SLAM already does this," respond: "agreed, that's why I'm proposing to subsume deformation under the unified lifecycle framework (slide 12) rather than treat it separately."

5. **Have the working code visible.** If a question gets technical, screen-share the GitHub repo. Demonstrating that you've actually implemented what you're describing builds credibility instantly.
