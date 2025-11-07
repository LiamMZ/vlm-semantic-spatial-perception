from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vlm-spatial-perception",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="VLM-powered semantic spatial perception with PDDL planning for robotics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vlm-spatial-perception",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "google-generativeai>=0.3.0",
        "pddl>=0.3.0",
        "scipy>=1.11.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "realsense": ["pyrealsense2>=2.55.0"],
        "visualization": [
            "matplotlib>=3.7.0",
            "plotly>=5.18.0",
            "open3d>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vlm-perception=main:main",
        ],
    },
)
