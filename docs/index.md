---
layout: default
title: Home
nav_order: 1
---

<img src="https://github.com/materialsvirtuallab/maml/blob/master/resources/logo_horizontal.png?raw=true" alt="maml" width="50%">

[![GitHub license](https://img.shields.io/github/license/materialsvirtuallab/maml)](https://github.com/materialsvirtuallab/maml/blob/main/LICENSE)
[![Linting](https://github.com/materialsvirtuallab/maml/workflows/Linting/badge.svg)](https://github.com/materialsvirtuallab/maml/workflows/Linting/badge.svg)
[![Testing](https://github.com/materialsvirtuallab/maml/workflows/Testing/badge.svg)](https://github.com/materialsvirtuallab/maml/workflows/Testing/badge.svg)
[![Downloads](https://pepy.tech/badge/maml)](https://pepy.tech/project/maml)
[![codecov](https://codecov.io/gh/materialsvirtuallab/maml/branch/master/graph/badge.svg?token=QNL1CRLVVL)](https://codecov.io/gh/materialsvirtuallab/maml)

maml (MAterials Machine Learning) is a Python package that aims to provide useful high-level interfaces that make ML
for materials science as easy as possible.

The goal of maml is not to duplicate functionality already available in other packages. maml relies on well-established
packages such as scikit-learn and tensorflow for implementations of ML algorithms, as well as other materials science
packages such as [pymatgen](http://pymatgen.org) and [matminer](http://hackingmaterials.lbl.gov/matminer/) for
crystal/molecule manipulation and feature generation.

Official documentation at https://materialsvirtuallab.github.io/maml/

# Features

1. Convert materials (crystals and molecules) into features. In addition to common compositional, site and structural
   features, we provide the following fine-grain local environment features.

 a) Bispectrum coefficients
 b) Behler Parrinello symmetry functions
 c) Smooth Overlap of Atom Position (SOAP)
 d) Graph network features (composition, site and structure)

2. Use ML to learn relationship between features and targets. Currently, the `maml` supports `sklearn` and `keras`
   models.

3. Applications:

 a) `pes` for modelling the potential energy surface, constructing surrogate models for property prediction.

  i) Neural Network Potential (NNP)
  ii) Gaussian approximation potential (GAP) with SOAP features
  iii) Spectral neighbor analysis potential (SNAP)
  iv) Moment Tensor Potential (MTP)

 b) `rfxas` for random forest models in predicting atomic local environments from X-ray absorption spectroscopy.

 c) `bowsr` for rapid structural relaxation with bayesian optimization and surrogate energy model.

# Installation

Pip install via PyPI:

```bash
pip install maml
```

To run the potential energy surface (pes), lammps installation is required you can install from source or from `conda`::

```bash
conda install -c conda-forge/label/cf202003 lammps
```

The SNAP potential comes with this lammps installation. The GAP package for GAP and MLIP package for MTP are needed to run the corresponding potentials. For fitting NNP potential, the `n2p2` package is needed.

Install all the libraries from requirements.txt file::

```bash
pip install -r requirements.txt
```

For all the requirements above:

```bash
pip install -r requirements-ci.txt
pip install -r requirements-optional.txt
pip install -r requirements-dl.txt
pip install -r requirements.txt
```

# Usage

Many Jupyter notebooks are available on usage. See [notebooks](https://github.com/materialsvirtuallab/maml/tree/master/notebooks). We also have a tool and tutorial lecture
at [nanoHUB](https://nanohub.org/resources/maml).

# API documentation

See [API docs](https://materialsvirtuallab.github.io/maml/maml.html).

# Citing

```txt
@misc{
    maml,
    author = {Chen, Chi and Zuo, Yunxing, Ye, Weike, Ji, Qi and Ong, Shyue Ping},
    title = {{Maml - materials machine learning package}},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/materialsvirtuallab/maml}},
}
```

For the ML-IAP package (`maml.pes`), please cite::

```txt
Zuo, Y.; Chen, C.; Li, X.; Deng, Z.; Chen, Y.; Behler, J.; Csányi, G.; Shapeev, A. V.; Thompson, A. P.;
Wood, M. A.; Ong, S. P. Performance and Cost Assessment of Machine Learning Interatomic Potentials.
J. Phys. Chem. A 2020, 124 (4), 731–745. https://doi.org/10.1021/acs.jpca.9b08723.
```

For the BOWSR package (`maml.bowsr`), please cite::

```txt
Zuo, Y.; Qin, M.; Chen, C.; Ye, W.; Li, X.; Luo, J.; Ong, S. P. Accelerating Materials Discovery with Bayesian
Optimization and Graph Deep Learning. Materials Today 2021, 51, 126–135.
https://doi.org/10.1016/j.mattod.2021.08.012.
```

For the AtomSets model (`maml.models.AtomSets`), please cite::

```txt
Chen, C.; Ong, S. P. AtomSets as a hierarchical transfer learning framework for small and large materials
datasets. Npj Comput. Mater. 2021, 7, 173. https://doi.org/10.1038/s41524-021-00639-w
```
