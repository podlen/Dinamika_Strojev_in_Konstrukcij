from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"


setup(
    name="fem_tools",
    version="0.1.0",
    description="Finite-element utilities for truss and frame dynamics.",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=["numpy", "scipy", "matplotlib"],
)