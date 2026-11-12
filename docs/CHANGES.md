# Change log

## v2026.11.12

### Breaking changes
* Python >= 3.11 is required. DGL is no longer a dependency; `matgl>=3.0.3` (PyTorch backend) is required.
* MatGL describers (`MatGLStructure`, `MatGLSite`) and `DIRECTSampler` default to `M3GNet-Eform-MP-2019.4.1`.
  The previous default, `M3GNet-Eform-MP-2018.6.1`, has been removed from Hugging Face. (#731)
* `MTPotential` defaults to MLIP-3 (`version="mlip-3"`). MLIP-2 and mlip-dev remain available via `version`. (#728)
* `ElementStats` weighted `std`, `skewness`, `kurtosis` and `moment:k` (k >= 2) change for compositions with
  unequal element fractions, after a fix to the weighted root moment. Retrain models built on these features. (#723)

### New features
* `MatGLStructure` and `MatGLSite` describers for MatGL foundation models.
* `MTPotential` supports MLIP-3 for `train()` and `evaluate()`. MLIP-3's LAMMPS interface is not yet supported. (#728)
* `MTPotential.train()` with MLIP-3 writes atomic numbers as atom types, so the fitted potential records its
  elements. `MTPotential.from_config()` reads them from the file and `elements` becomes optional. (#618)
* `MTPotential.from_config()` validates `elements` against `species_count`, rejects duplicates and warns when the
  order differs from the electronegativity order used in training. `default_element_ordering=True` reorders. (#618)
* `BirchClustering` uses `fastcluster` for the Ward step when `n` is set, cutting memory from O(M^2) to O(M·d) for
  large DIRECT samples. `use_fastcluster=False` restores the scikit-learn step. (#732)

### Bug fixes
* MTP: `min_dist` is computed in Python instead of `mlp mindist` (#601), and a diverged fit raises a clear error
  instead of `JSONDecodeError` (#702). (#728)
* GAP: `dft_virial` is quoted in extended XYZ files, so `gap_fit` trains with stresses and `evaluate()` reads
  `quip` output. `gap_fit`/`quip` failures report the program's own error. (#729, #677)
* Describers: `n_jobs` and `memory` work for model-backed describers (M3GNet, MatGL). (#730, #637)
* n2p2: NaN-padded columns are filtered when parsing `elements` and `global_nodes_short` in `load_input`.
* 42 bugs found while raising test coverage: `maml.apps.pes` (15), core modules (9), `symbolic` (7),
  `describers` (5), `bowsr` (5) and `gbe` (`GBBond` break-bond ratio). (#722-#727)

### Other
* Test coverage raised to ~99% for most modules, with mocked binaries for the PES apps and a real MLIP-3 binary in CI.
* README rewritten; notebooks and docs updated.

## 2024.6.13
* Misc bug and doc fixes.
* Code cleanup.

## v2023.9.9
* Added DIRECT sampling approach in `maml.sampling`.

## v2022.9.20
* Bug fix for BOWSR failing because of indexing in some instances.

## v2021.1.28
* Fix SNAP stress components
* Fix unexpected test outcome in BOWSR and symbolic regression

## v2021.12.13

* Update dependencies
* Fix bugs

## v2021.10.13
* add bowsr

## v2021.3.2
* Set up namespace package architecture for maml.apps. The RFXAS package is now available as a separate package
  `maml-apps.rfxas`. Usage remains similar.

## v2021.2.27
* maml is now compatible with both mlip-2 and mlip-dev

## v0.0.4
* bug fix
* compatibility with new n2p2 and mlip software
* upgrade rfxas

## v0.0.3
* bug fix
* add atomsets models

## v0.0.2
* major refactor
* move mlearn potential energy surfaces to apps
* fix potential problem of rotated structure in lammps
* add rfxas app
* add initial symbolic regression app
* add megnet-based describers

## v0.0.1
* initial release
