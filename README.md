# `madjax` - differentiable HEP Matrix Elements

One of the most important quantities in HEP are the gradients of the
probabilities of a given particle collision final state ("Matrix Elments")
with respect to theory parameters or phase-space coordinates.

The `madjax` package aims to provide these in an easy to use Python-focused
framework through the use of the automatic differentiation `jax` and by integrating
with the Matrix Element Calculator [`MadGraph`][MadGraph5_website]. It consists of two modules

* a `MadGraph` plugin to generate differentiable code
* a Python module `madjax` that provides an easy to use interface

## Installation

To get `madjax` you can get the latest release [from PyPI][madjax_PyPI] with

```console
python -m pip install madjax
```

`madjax` relies on `MadGraph` for code generation using its plugin. The recommended
way to do this is through the use of official `madjax` docker images

```console
docker pull madjax/madjax
```

It is also possible to use `madjax` with a local `MadGraph` install.
You just need to copy the contents of `$(madjax-config)` into the MadGraph installation's `PLUGIN/` directory

```console
cp -r "$(madjax-config)" <path to MadGraph5 install directory>/PLUGIN/
```

If you only have one version of `MadGraph` installed on your machine, the following _may_ work

```console
cp -r "$(madjax-config)" "$(grep mg5_path $(find / -type f -iname mg5_configuration.txt) | head -n 1 | awk '{print $(NF)}')/PLUGIN/"
```

## Usage

[MadGraph5_website]: https://launchpad.net/mg5amcnlo
[madjax_PyPI]: https://pypi.org/project/madjax/
