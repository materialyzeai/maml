<img src="https://github.com/materialyzeai/maml/blob/main/resources/logo_horizontal.png?raw=true" alt="maml" width="50%">

[![GitHub license](https://img.shields.io/github/license/materialyzeai/maml)](https://github.com/materialyzeai/maml/blob/main/LICENSE)
[![Linting](https://github.com/materialyzeai/maml/actions/workflows/lint.yml/badge.svg)](https://github.com/materialyzeai/maml/actions/workflows/lint.yml)
[![Testing](https://github.com/materialyzeai/maml/actions/workflows/test.yml/badge.svg)](https://github.com/materialyzeai/maml/actions/workflows/test.yml)
[![Downloads](https://static.pepy.tech/badge/maml)](https://pepy.tech/projects/maml)
[![codecov](https://codecov.io/gh/materialyzeai/maml/branch/main/graph/badge.svg)](https://codecov.io/gh/materialyzeai/maml)

maml (MAterials Machine Learning) provides high-level interfaces that make machine learning for materials science as
easy as possible. It does not reimplement ML algorithms. Instead, it builds on [scikit-learn](https://scikit-learn.org),
[TensorFlow](https://www.tensorflow.org) and [MatGL](https://matgl.ai) for models, and on
[pymatgen](https://pymatgen.org) and [matminer](https://hackingmaterials.lbl.gov/matminer/) for structure handling and
featurization.

- Documentation: https://materialyzeai.github.io/maml/
- API reference: https://materialyzeai.github.io/maml/maml.html
- Example notebooks: [notebooks](https://github.com/materialyzeai/maml/tree/main/notebooks)
- Tutorial lecture and hosted tool: [nanoHUB](https://nanohub.org/resources/maml)

# Features

| Module | What it does |
| --- | --- |
| `maml.describers` | Convert compositions, sites and structures into features: element statistics, Coulomb matrices, radial distribution functions, bispectrum coefficients, Behler-Parrinello symmetry functions, SOAP, and graph-network (MatGL, MEGNet, M3GNet) embeddings. Any matminer featurizer can be wrapped with `wrap_matminer_describer`. |
| `maml.models` | Describer-aware wrappers around scikit-learn (`SKLModel`) and Keras (`KerasModel`, `MLP`) models, plus the `AtomSets` transfer-learning architecture. |
| `maml.sampling` | DIRECT (DImensionality REduction-Clustering-sTratified) sampling for selecting robust, diverse training sets from large configuration spaces. |
| `maml.apps.pes` | Fit and evaluate machine-learning interatomic potentials: SNAP/qSNAP, GAP, MTP and NNP, with LAMMPS-based property calculators (elastic constants, lattice constants, defect formation, NEB). |
| `maml.apps.bowsr` | BOWSR: Bayesian optimization with a surrogate energy model for DFT-free structure relaxation. |
| `maml.apps.gbe` | Machine-learning model for elemental grain boundary energies. |
| `maml.apps.symbolic` | Symbolic regression and sparse feature selection: SIS/ISIS, LASSO, adaptive LASSO, SCAD, Dantzig selector and L0 search. |

# Installation

```bash
pip install maml
```

Or, with [uv](https://docs.astral.sh/uv/):

```bash
uv add maml
```

Optional extras:

```bash
pip install "maml[symbolic]"  # cvxpy, for maml.apps.symbolic
pip install "maml[deep]"      # TensorFlow, for Keras models and AtomSets
```

`maml.apps.pes` drives external codes, which must be installed separately and be on your `PATH`:

| Potential | Required code |
| --- | --- |
| All potentials (property calculations) | [LAMMPS](https://www.lammps.org) (`conda install -c conda-forge lammps`) |
| SNAP / qSNAP | LAMMPS (built in) |
| GAP | [QUIP](https://github.com/libAtoms/QUIP) with the GAP package, and LAMMPS built with `ML-QUIP` |
| MTP | [MLIP-3](https://gitlab.com/ashapeev/mlip-3) (default; MLIP-2 via `MTPotential(version="mlip-2")`). LAMMPS-based MTP calculations currently require MLIP-2 and its LAMMPS interface |
| NNP | [n2p2](https://github.com/CompPhysVienna/n2p2) |

## Development

```bash
git clone https://github.com/materialyzeai/maml.git
cd maml
uv sync --group lint
uv run pytest tests
```

# Quick start

Train a composition model with scikit-learn:

```python
from pymatgen.core import Composition
from sklearn.linear_model import LinearRegression

from maml.describers import ElementStats
from maml.models import SKLModel

describer = ElementStats.from_data("megnet_3", stats=["mean", "std"])
model = SKLModel(describer=describer, model=LinearRegression())

compositions = [Composition(c) for c in ["Li2O", "Na2O", "K2O", "MgO", "CaO"]]
targets = [-2.07, -1.40, -1.13, -3.08, -3.27]  # formation energies, eV/atom
model.train(compositions, targets)
print(model.predict_objs([Composition("SrO")]))
```

See the [notebooks](https://github.com/materialyzeai/maml/tree/main/notebooks) for worked examples of each module.

# Citing

If you use maml, please cite:

```bibtex
@misc{maml,
    author = {Chen, Chi and Zuo, Yunxing and Ye, Weike and Qi, Ji and Ong, Shyue Ping},
    title = {{maml - materials machine learning package}},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/materialyzeai/maml}},
}
```

Please also cite the paper for the module you use:

- **ML-IAPs (`maml.apps.pes`)**: Zuo, Y.; Chen, C.; Li, X.; Deng, Z.; Chen, Y.; Behler, J.; Csányi, G.; Shapeev,
  A. V.; Thompson, A. P.; Wood, M. A.; Ong, S. P. Performance and Cost Assessment of Machine Learning Interatomic
  Potentials. *J. Phys. Chem. A* **2020**, 124 (4), 731–745. https://doi.org/10.1021/acs.jpca.9b08723
- **BOWSR (`maml.apps.bowsr`)**: Zuo, Y.; Qin, M.; Chen, C.; Ye, W.; Li, X.; Luo, J.; Ong, S. P. Accelerating
  Materials Discovery with Bayesian Optimization and Graph Deep Learning. *Mater. Today* **2021**, 51, 126–135.
  https://doi.org/10.1016/j.mattod.2021.08.012
- **AtomSets (`maml.models.AtomSets`)**: Chen, C.; Ong, S. P. AtomSets as a Hierarchical Transfer Learning Framework
  for Small and Large Materials Datasets. *npj Comput. Mater.* **2021**, 7, 173.
  https://doi.org/10.1038/s41524-021-00639-w
- **DIRECT sampling (`maml.sampling`)**: Qi, J.; Ko, T. W.; Wood, B. C.; Pham, T. A.; Ong, S. P. Robust Training of
  Machine Learning Interatomic Potentials with Dimensionality Reduction and Stratified Sampling. *npj Comput. Mater.*
  **2024**, 10, 43. https://doi.org/10.1038/s41524-024-01227-4
- **Grain boundary energies (`maml.apps.gbe`)**: Ye, W.; Zheng, H.; Chen, C.; Ong, S. P. A Universal Machine Learning
  Model for Elemental Grain Boundary Energies. *Scr. Mater.* **2022**, 218, 114803.
  https://doi.org/10.1016/j.scriptamat.2022.114803
