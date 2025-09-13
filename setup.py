from setuptools import setup, find_packages

setup(
    name="temporal-multimodal-embodied-ai",
    version="0.1.0",
    description="Temporal reasoning capabilities for vision-language models in embodied AI",
    author="Your Name",
    author_email="your.email@university.edu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "habitat-sim>=0.2.3",
        "opencv-python>=4.5.0",
        "requests>=2.28.0",
        "numpy>=1.21.0",
        "pillow>=8.3.0",
        "torch>=1.12.0",
        "transformers>=4.20.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)