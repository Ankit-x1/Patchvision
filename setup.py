from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies required for the library to function
install_requires = [
    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
    "pandas>=2.0.0",
    "py-cpuinfo>=9.0.0",
    "psutil>=5.9.0",
]

# Optional dependencies, grouped by functionality
extras_require = {
    "torch": ["torch>=2.0.0", "torchvision>=0.15.0"],
    "gpu": ["cupy-cuda11x>=12.0.0"],  # For CUDA accelerated array operations
    "deploy": [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "websockets>=12.0",
        "onnxruntime>=1.16.0",
    ],
    "viz": [
        "plotly>=5.17.0",
        "dash>=2.14.0",
        "dash-bootstrap-components>=1.4.2",
        "kaleido>=0.2.0",
    ],
    "tuning": ["optuna>=3.0.0"],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
    ],
}

# Create a 'full' extra that includes all optional dependencies for convenience
extras_require["full"] = (
    extras_require["torch"]
    + extras_require["gpu"]
    + extras_require["deploy"]
    + extras_require["viz"]
    + extras_require["tuning"]
)


setup(
    name="patchvision",
    version="1.0.0",
    author="Ankit Karki",
    author_email="karkiankit101@gmail.com",
    description="Industrial-grade vision processing framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ankit-x1/Patchvision",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "patchvision=patchvision.main:main",
        ],
    },
    include_package_data=True,
)