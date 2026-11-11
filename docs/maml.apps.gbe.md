---
layout: default
title: maml.apps.gbe.md
nav_exclude: true
---

# maml.apps.gbe package

Implementation of GB energy prediction paper
Weike Ye, Hui Zheng, Chi Chen, Shyue Ping Ong
“A Universal Machine Learning Model for Elemental Grain Boundary Energies”
arXiv preprint arXiv: 2201.11991 (2022).

## maml.apps.gbe.describer module

Module implements the describers for GB entry.

### *class* maml.apps.gbe.describer.GBBond(gb: GrainBoundary, loc_algo: str = ‘crystalnn’, bond_mat: np.ndarray | None = None, \*\*kwargs)

Bases: `MSONable`

This class describes the GB Bonding environment using provided
local environment algorithm.
Available algorithms: GBBond.NNDict.keys(), default CrystalNN.

#### NNDict(_ = {‘brunnernn_real’: <class ‘pymatgen.analysis.local_env.BrunnerNN_real’>, ‘brunnernn_reciprocal’: <class ‘pymatgen.analysis.local_env.BrunnerNN_reciprocal’>, ‘brunnernn_relative’: <class ‘pymatgen.analysis.local_env.BrunnerNN_relative’>, ‘covalentbondnn’: <class ‘pymatgen.analysis.local_env.CovalentBondNN’>, ‘critic2nn’: <class ‘pymatgen.analysis.local_env.Critic2NN’>, ‘crystalnn’: <class ‘pymatgen.analysis.local_env.CrystalNN’>, ‘cutoffdictnn’: <class ‘pymatgen.analysis.local_env.CutOffDictNN’>, ‘econnn’: <class ‘pymatgen.analysis.local_env.EconNN’>, ‘jmolnn’: <class ‘pymatgen.analysis.local_env.JmolNN’>, ‘minimumdistancenn’: <class ‘pymatgen.analysis.local_env.MinimumDistanceNN’>, ‘minimumokeeffenn’: <class ‘pymatgen.analysis.local_env.MinimumOKeeffeNN’>, ‘minimumvirenn’: <class ‘pymatgen.analysis.local_env.MinimumVIRENN’>, ‘nearneighbors’: <class ‘pymatgen.analysis.local_env.NearNeighbors’>, ‘openbabelnn’: <class ‘pymatgen.analysis.local_env.OpenBabelNN’>, ‘voronoinn’: <class ‘pymatgen.analysis.local_env.VoronoiNN’>_ )

#### \_get_bond_mat(gb: GrainBoundary)

* **Parameters**
  **gb** (*GrainBoundary*) – the grain boundary structure object.
* **Returns**
  bond matrix
  matrix of bond lengths
  1. bm[i][j] = bond length between atom i&j
     if i & j is bonded (determined by the loc_algo)
  2. bm[i][j] = bm[j][i]
  3. If not bonded, the bm[i][j] = np.nan

#### as_dict()

Dict representation of the GBond class.

* **Returns**
  str, “bond_mat”: bond matrix}
* **Return type**
  dict of {“loc_algo”

#### *property* bond_matrix(    *: ndarra* )

The (padded) bond matrix.

#### *classmethod* from_dict(d)

* **Parameters**
  **d** (*dict*) – Dict representation.
* **Returns**
  GBBond

#### get_breakbond_ratio(gb: GrainBoundary, b0: float, return_count: bool = False)

Get the breakbond ratio, i.e the ratio of shorter bonds vs longer bonds
compared to the bulk bond length (b0)
The algo to find the bonds can vary
Meme: if use get_neighbors, require a hard set cutoff, which adds to

> arbitrariness.
* **Parameters**
  * **gb** (*GrainBoundary*) – a GrainBoundary object
  * **b0** (*float*) – cutoff to determine short vs. long bonds,
    default the bulk bond length
  * **return_count** (*bool*) – whether to return count of
* **Returns**
  shorter_bond / longer_bonds
  if return_count:
  shorter_bond: # of short bonds
  longer_bond: # of long bonds
* **Return type**
  ratio (float)

#### get_mean_bl_chg(b0: float)

Function to calculate the mean bond length difference between GB and the bulk
:param b0: the bond length in bulk.
:type b0: float

Returns: the mean_bl_chg

#### *property* max_bl(    *: floa* )

The maximum bond length.

#### *property* min_bl(    *: floa* )

The minimum bond length.

### *class* maml.apps.gbe.describer.GBDescriber(structural_features: list | None = None, elemental_features: list | None = None, \*\*kwargs)

Bases: `BaseDescriber`

The describers that describes the grain boundary db entry
with selected structural and elemental features.

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### generate_bulk_ref(gb_df: DataFrame, inc_target: bool = True)

Generate the bulk reference for given gb entry
:param gb_df: data for gb
:type gb_df: pd.DataFrame
:param inc_target: whether to include target.
:type inc_target: bool

* **Returns**
  data for bulk
* **Return type**
  bulk_df (pd.DataFrame)

#### transform_one(db_entry: dict, inc_target: bool = True, inc_bulk_ref: bool = True, mp_api: str | None = None)

Describe gb with selected structural and elemental features
:param db_entry: entry from surfacedb.All_GB_properties_parallel_copy
:type db_entry: dict
:param inc_target: whether to include target in the df, default: False
:type inc_target: bool
:param inc_bulk_ref: whether to generate bulk reference

> bulk reference: i.e. the entry of the origin bulk of the GB,
> ```none
> the rotation angle (theta) = 0, gb_energy = 0
> ```
* **Parameters**
  **mp_api** (*str*) – MP api key.

Returns: pd.DataFrame of processed data, columns are the feature labels

```none
The task_id is included and serve as the unique id of the data
```

### maml.apps.gbe.describer.convert_hcp_direction(rotation_axis: list, lat_type: str)

four index notion to three index notion for hcp and rhombohedral axis
:param rotation_axis: four index notion of axis
:type rotation_axis: list
:param lat_type: the
:type lat_type: str

* **Returns**
  rotation axis in three index notion.

### maml.apps.gbe.describer.convert_hcp_plane(plane: list)

four index notion to three index notion for hcp and rhombohedral plane
:param plane: four index notion.
:type plane: list

* **Returns**
  three index notion of plane

### maml.apps.gbe.describer.get_elemental_feature(db_entry: dict, loc_algo: str = ‘crystalnn’, features: list | None = None, mp_api: str | None = None)

Function to get the elemental features
:param db_entry: db entry
:type db_entry: Dict
:param loc_algo: algorithm to determine local env.

> Options: see GBBond.NNDict.keys()
* **Parameters**
  * **features** (*List*) – list of feature names
    e_coh: cohesive energy
    G: G_vrh shear modulus
    a0: bulk lattice parameter a
    ar: atomic radius
    b0: the bond length of the metal bulk
    mean_delta_bl: the mean bond length difference

> between GB and the bulk

bl: the mean bond length in GB

```none
* **mp_api** (*str*) – api key to MP.
```

* **Returns**
  pd.DataFrame of elemental features

### maml.apps.gbe.describer.get_structural_feature(db_entry: dict, features: list | None = None)

The structural features:
d_gb: interplanal distance of the gb_plane
d_rot: interplanal distance of the gb_plane

> w/ smallest integer index ) normal to rotation axis

theta: rotation angle (sin and cos).

* **Parameters**
  * **db_entry** (*Dict*) – db entry
  * **features** (*List*) – list of features.
* **Returns**
  pd.DataFrame of structural features

## maml.apps.gbe.presetfeatures module

Module defines feature objects for GB energy model.

### *class* maml.apps.gbe.presetfeatures.my_quant(str_name: str, latex_name: str | None = None, latex_unit: str | None = None)

Bases: `object`

An object to describe GB quantities.

#### *property* latex()

Returns:
latex rep of the quant.

#### *property* name()

Returns:
string rep of the quant.

#### *property* unit()

Returns:
latex rep of the quant unit.

## maml.apps.gbe.utils module

Module implements helper functions to retrieve data for GB energy prediction paper.

### maml.apps.gbe.utils.load_b0_dict()

Helper function to retrieve the b0 data.
b0 is the bulk bond length.

* **Returns**
  b0}
* **Return type**
  returns the dict of {element

### maml.apps.gbe.utils.load_data(filename: str | None = None)

Helper function to load the data
Default is to load the 361 data
:param filename: the filename of the data.
:type filename: str

* **Returns**
  data list

### maml.apps.gbe.utils.load_mean_delta_bl_dict(loc_algo: str = ‘crystalnn’)

Helper function to load the mean_delta_bl data
:param loc_algo: name of the algorithm.
:type loc_algo: str

* **Returns**
  mean_delta_bl(float)}
* **Return type**
  {task_id (int)

### maml.apps.gbe.utils.update_b0_dict(data_source: list, mp_api: str | None)

Helper function to update the b0 dictionary
Requires api key to MP
:param data_source: list of GB data
:type data_source: list
:param mp_api: API key to MP
:type mp_api: str

* **Returns**
  {el: b0}.
* **Return type**
  b0_dict (dict)