import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="punchbowl",
    version="0.0.1",
    packages=find_packages(),
    url="",
    license="",
    author="PUNCH SOC",
    author_email="mhughes@boulder.swri.edu",
    description="PUNCH science calibration code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        "numpy",
        "astropy",
        "sunpy",
        "pandas",
        "ndcube",
        "matplotlib",
        "ccsdspy",
        "prefect",
        "regularizepsf",
        "solpolpy",
        "scipy",
        "lmfit",
        "sunkit-image",
        "setuptools",
        "reproject",
        "pylibjpeg",
        "python-dateutil",
        "pyyaml"
    ],
    extras_require={
        "dev": ["pytest",
                "coverage",
                "pytest-cov",
                "flake8",
                "pytest-runner",
                "pytest-mpl",
                "pre-commit",
                "ruff",
                "hypothesis"],
        "docs": ["sphinx",
                 "pydata-sphinx-theme",
                 "sphinx-autoapi",
                 "sphinx-favicon",
                 "nbsphinx",
                 "ipython",
                 "sphinxcontrib-mermaid"],
    },
)
