from setuptools import setup, find_packages

setup(
    name="blt",
    version="0.1.0",
    description="Byte Latent Transformer (BLT): Patches Scale Better Than Tokens",
    author="BLT Project Contributors",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "black", "ruff"],
        "eval": ["tokenizers>=0.15.0", "tiktoken>=0.6.0", "rapidfuzz>=3.6.0", "sacrebleu>=2.4.0"],
        "accel": ["numba>=0.58.0", "bitsandbytes"],
    },
)
