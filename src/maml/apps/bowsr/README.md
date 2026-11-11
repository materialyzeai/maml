## Bayesian Optimization With Symmetry Relaxation Algorithm (BOWSR)

The BOWSR app is designed for obtaining near-equilibrium crystal structures via
Bayesian optimization and graph deep learning energy model without expensive
DFT. The current implementation is compatible with energy evaluators
for [MEGNet](https://github.com/materialsvirtuallab/megnet)
, [CGCNN](https://github.com/txie-93/cgcnn), and [VASP](https://www.vasp.at).

# Algorithm

The BOWSR algorithm parametrizes each crystal based on the independent lattice
parameters and atomic coordinates based on its space group. The potential
energy surface is then approximated by initializing a set of training
observations and energies from the ML energy model. Bayesian optimization is
then used to iteratively propose lower energy geometries based on prior
observations.

![BOWSR algorithm](../../../resources/bowsr_algo.png)
<div align='center'><strong>Figure 1. Bayesian Optimization With Symmetry Relaxation (BOWSR) algorithm.</strong></div>

# Usage

By default, the current implementation uses the pre-trained graph models
in [MEGNet-2019](https://github.com/materialsvirtuallab/megnet/tree/master/mvl_models/mp-2019.4.1)
and [CGCNN-2018](https://github.com/txie-93/cgcnn/tree/master/pre-trained).
Please visit the [notebooks directory](../../../notebooks/bowsr/bowsr) for Jupyter
notebooks with more detailed code example.

|                                       |                                                                                                                      |                                                                                                                   |                                                                                                                                               |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **BOWSR with MEGNet as energy model** | [![Binder]](https://mybinder.org/v2/gh/materialsvirtuallab/maml/master?labpath=notebooks/bowsr/megnet_example.ipynb) | [![View on GitHub]](https://github.com/materialsvirtuallab/maml/blob/master/notebooks/bowsr/megnet_example.ipynb) | [![Open in Google Colab]](https://colab.research.google.com/github/materialsvirtuallab/maml/blob/master/notebooks/bowsr/megnet_example.ipynb) |
| **BOWSR with CGCNN as energy model**  | [![Binder]](https://mybinder.org/v2/gh/materialsvirtuallab/maml/master?labpath=notebooks/bowsr/cgcnn_example.ipynb)  | [![View on GitHub]](https://github.com/materialsvirtuallab/maml/blob/master/notebooks/bowsr/cgcnn_example.ipynb)  | [![Open in Google Colab]](https://colab.research.google.com/github/materialsvirtuallab/maml/blob/master/notebooks/bowsr/cgcnn_example.ipynb)  |

[Binder]: https://mybinder.org/badge_logo.svg
[View on GitHub]: https://img.shields.io/badge/View%20on-GitHub-darkblue?logo=github
[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg

## Dependencies

Upgrade pip and install dependencies.

```python
pip install --upgrade pip
pip install maml megnet tensorflow
```

## Minimal Usage

### Imports

Import the necessary modules:

```python
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import get_el_sp
from maml.apps.bowsr.model.megnet import MEGNet

# from maml.apps.bowsr.model.cgcnn import CGCNN
from maml.apps.bowsr.optimizer import BayesianOptimizer
```

## MEGNet, Pymatgen, and Helper Function

Instantiate a MEGNet (or CGCNN) model and load a `pymatgen` `Structure` from a CIF file:

```python
model = MEGNet()  # or `model = CGCNN()`
structure = Structure.from_file("<my-file>.cif")
```

We use a helper function for the expected radius based on the sum of radii of two elements:

```python
def expected_cutoff(struct):
    if struct.composition.num_atoms == 1:
        return 2 * struct.composition.elements[0].atomic_radius
    return sum([e.atomic_radius for e in sorted(struct.composition.elements, key=lambda x: x.atomic_radius)[:2]])
```

## Optimization

Finally, instantiate and run the optimizer:

```python
compressed_optimizer = BayesianOptimizer(
    model=model, structure=structure, relax_coords=True, relax_lattice=True, use_symmetry=True, seed=42
)

compressed_optimizer.set_bounds()
compressed_optimizer.optimize(n_init=100, n_iter=100, alpha=0.026**2)
```

## Relaxed Structure

To retrieve the relaxed structure:

```python
cutoff_distance = max(round(expected_cutoff(compressed_optimizer.structure) * 0.6, 2), 1.1)
relaxed, _ = compressed_optimizer.get_optimized_structure_and_energy(cutoff_distance=cutoff_distance)
print(relaxed)
```

```txt
Full Formula (Sr1 Ti1 O3)
Reduced Formula: SrTiO3
abc   :   3.976834   3.976834   3.976834
angles:  90.000000  90.000000  90.000000
Sites (5)
  #  SP      a    b    c
---  ----  ---  ---  ---
  0  Sr    0    0    0
  1  Ti    0.5  0.5  0.5
  2  O     0.5  0    0.5
  3  O     0    0.5  0.5
  4  O     0.5  0.5  0
```
