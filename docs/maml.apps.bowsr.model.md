---
layout: default
title: maml.apps.bowsr.model.md
nav_exclude: true
---

# maml.apps.bowsr.model package

Energy surrogate model.

## *class* maml.apps.bowsr.model.EnergyModel()

Bases: `object`

Base energy model class. For any model used in BOWSR, it has to have
a predict_energy method that returns a float.

### predict_energy(structure: Structure)

Predict the energy of a structure
:param structure: Pymatgen Structure object.

Returns: (float) energy value.

## maml.apps.bowsr.model.base module

Base class that expose a predict_energy method.

### *class* maml.apps.bowsr.model.base.EnergyModel()

Bases: `object`

Base energy model class. For any model used in BOWSR, it has to have
a predict_energy method that returns a float.

#### predict_energy(structure: Structure)

Predict the energy of a structure
:param structure: Pymatgen Structure object.

Returns: (float) energy value.

## maml.apps.bowsr.model.cgcnn module

CGCNN Wrapper.

## maml.apps.bowsr.model.dft module

DFT wrapper.

### *class* maml.apps.bowsr.model.dft.DFT(exe_path: str | None = None)

Bases: `EnergyModel`

DFT static calculation wrapped as energy model.

#### predict_energy(structure: Structure)

Predict energy from structure.

* **Parameters**
  **structure** – (pymatgen Structure).

Returns: float

## maml.apps.bowsr.model.megnet module

megnet model wrapper implementation.