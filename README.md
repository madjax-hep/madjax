# `madjax` - differentiable HEP Matrix Elements

One of the most important quantities in HEP are the gradients of the
probabilties of a given particle collision final state ("Matrix Elments")
with respect to theory parameters or phase-space coordinates.

The `madjax` package aims to provide these in an easy to use python-focused
framework through the use of the automatic differrention `jax` and by integrating
with the Matrix Element Calculator `MadGraph`. It consists of two modules

* a `Madgraph` plugin to generate differentiable code
* a python module `madjax` that provides an easy to use interface

## Installation

To get madgraph you can get the latest release through a simple `pip install`

```
pip install madjax
```

`madjax` relies on `MadGraph` for code generation using its plugin. The recommended
way to do this is through the use of official `madjax` docker images

```
docker pull madjax/madjax
```

but it's also possible to use `madjax` with a local `MadGraph` install.


## Usage
