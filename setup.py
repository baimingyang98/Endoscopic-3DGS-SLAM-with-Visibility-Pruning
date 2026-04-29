from setuptools import setup, find_packages

setup(
    name="endogslam-innovations",
    version="1.0.0",
    description="Robust Gaussian Map Management for Endoscopic SLAM: "
                "Visibility-Guided Pruning and Bundle Adjustment",
    author="Bai Mingyang",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.5",
        "tqdm>=4.65.0",
        "opencv-python>=4.9.0",
        "Pillow>=9.2.0",
        "matplotlib>=3.5.2",
        "kornia",
        "pyyaml",
        "lpips",
        "pytorch-msssim",
        "torchmetrics",
    ],
)
