"""Setup madjax."""
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as readme_md:
    long_description = readme_md.read()

extras_require = {}
extras_require["test"] = sorted(
    set(
        [
            "pyflakes",
            "pytest~=3.5",
            "pydocstyle",
            "papermill~=1.0",
            'black;python_version>="3.6"',  # Black is Python3 only
        ]
    )
)
extras_require["docs"] = sorted(
    set(
        [
            "sphinx",
            "sphinxcontrib-bibtex",
            "sphinx-click",
            "sphinx_rtd_theme",
            "nbsphinx",
            "ipywidgets",
            "sphinx-issues",
            "m2r",
        ]
    )
)
extras_require["develop"] = sorted(
    set(
        extras_require["docs"]
        + extras_require["test"]
        + ["nbdime", "bumpversion", "ipython", "pre-commit", "twine", "pydocstyle"]
    )
)
extras_require["complete"] = sorted(set(sum(extras_require.values(), [])))


setup(
    name="madjax",
    version="0.0.1",
    description="differentiable matrix elements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lukasheinrich/madjax",
    author="Lukas Heinrich, Michael Kagan",
    author_email="lukas.heinrich@cern.ch, mkagan@cern.ch",
    license="Apache",
    keywords="physics autodifferentiation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=["jax~=0.1,>0.1.51", "jaxlib~=0.1,>0.1.33"],
    extras_require=extras_require,
    dependency_links=[],
    use_scm_version=lambda: {"local_scheme": lambda version: ""},
)
