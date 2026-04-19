from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).resolve().parent
README_PATH = BASE_DIR / "README.md"
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"


def read_readme() -> str:
    if README_PATH.exists():
        return README_PATH.read_text(encoding="utf-8")
    return "Kidney Disease Classification using a custom CNN model."


def read_requirements() -> list[str]:
    if not REQUIREMENTS_PATH.exists():
        return []

    requirements: list[str] = []
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if requirement and not requirement.startswith("#"):
            requirements.append(requirement)
    return requirements


setup(
    name="kidney-disease-classifier",
    version="0.1.0",
    description="Production-ready kidney disease classification project using a custom CNN model.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="ASUS",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10,<3.12",
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
