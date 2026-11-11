---
layout: default
title: maml.describers.md
nav_exclude: true
---

# maml.describers package

Describer for converting (structural) objects into models-readable
numeric vectors or tensors.

## *class* maml.describers.BPSymmetryFunctions(cutoff: float, r_etas: ndarray, r_shift: ndarray, a_etas: ndarray, zetas: ndarray, lambdas: ndarray, feature_batch: str = ‘pandas_concat’, \*\*kwargs)

Bases: `BaseDescriber`

Behler-Parrinello symmetry function to describe the local environment
of each atom.

Reference:
@article{behler2007generalized,

> title={Generalized neural-network representation of
> ```none
> high-dimensional potential-energy surfaces},
> ```
> author={Behler, J{“o}rg and Parrinello, Michele},
> journal={Physical review letters},
> volume={98},
> number={14},
> pages={146401},
> year={2007},
> publisher={APS}}

### *abc_impl(* = <_abc.*abc_data object* )

### \_fc(r: float)

Cutoff function to decay the symmetry functions at vicinity of radial cutoff.

* **Parameters**
  **r** (*float*) – The pair distance.

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘site_ )

### transform_one(structure: Structure)

* **Parameters**
  **structure** (*Structure*) – Pymatgen Structure object.

## *class* maml.describers.BispectrumCoefficients(rcutfac: float, twojmax: int, element_profile: dict, quadratic: bool = False, pot_fit: bool = False, include_stress: bool = False, feature_batch: str = ‘pandas_concat’, \*\*kwargs)

Bases: `BaseDescriber`

Bispectrum coefficients to describe the local environment of each atom.
Lammps is required to perform this computation.

Reference:
@article{bartok2010gaussian,

> title={Gaussian approximation potentials: The
> ```none
> accuracy of quantum mechanics, without the electrons},
> ```
> author={Bart{‘o}k, Albert P and Payne, Mike C
> ```none
> and Kondor, Risi and Cs{'a}nyi, G{'a}bor},
> ```
> journal={Physical review letters},
> volume={104}, number={13}, pages={136403}, year={2010}, publisher={APS}}

### *abc_impl(* = <_abc.*abc_data object* )

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘site_ )

### *property* feature_dim(    *: int | Non* )

Bispectrum feature dimension.

### *property* subscripts(    *: lis* )

The subscripts (2j1, 2j2, 2j) of all bispectrum components
involved.

### transform_one(structure: Structure)

* **Parameters**
  **structure** (*Structure*) – Pymatgen Structure object.

## *class* maml.describers.CoulombEigenSpectrum(max_atoms: int | None = None, \*\*kwargs)

Bases: `BaseDescriber`

Get the Coulomb Eigen Spectrum describers.

Reference:
@article{rupp2012fast,

> title={Fast and accurate modeling of molecular
> ```none
> atomization energies with machine learning},
> ```
> author={Rupp, Matthias and Tkatchenko, Alexandre and M{“u}ller,
> ```none
> Klaus-Robert and Von Lilienfeld, O Anatole},
> ```
> journal={Physical review letters}, volume={108},
> number={5}, pages={058301},
> year={2012}, publisher={APS}}

### *abc_impl(* = <_abc.*abc_data object* )

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘structure_ )

### transform_one(mol: Molecule)

* **Parameters**
  **mol** (*Molecule*) – pymatgen molecule.

Returns: np.ndarray the eigen value vectors of Coulob matrix

## *class* maml.describers.CoulombMatrix(random_seed: int | None = None, max_atoms: int | None = None, is_ravel: bool = True, \*\*kwargs)

Bases: `BaseDescriber`

Coulomb Matrix to describe structure.

Reference:
@article{rupp2012fast,

> title={Fast and accurate modeling of molecular
> ```none
> atomization energies with machine learning},
> ```
> author={Rupp, Matthias and Tkatchenko, Alexandre and M{“u}ller,
> ```none
> Klaus-Robert and Von Lilienfeld, O Anatole},
> ```
> journal={Physical review letters}, volume={108},
> number={5}, pages={058301},
> year={2012}, publisher={APS}}

### *abc_impl(* = <_abc.*abc_data object* )

### *static* \_get_columb_mat(s: Molecule | Structure)

* **Parameters**
  **s** (*Molecule/Structure*) – input Molecule or Structure. Structure
  is not advised since the feature will depend on the supercell size.
* **Returns**
  Coulomb matrix of the structure

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘structure_ )

### get_coulomb_mat(s: Molecule | Structure)

* **Parameters**
  **s** (*Molecule/Structure*) – input Molecule or Structure. Structure
  is not advised since the feature will depend on the supercell size
* **Returns**
  Coulomb matrix of the structure.

### transform_one(s: Molecule | Structure)

* **Parameters**
  **s** (*Molecule/Structure*) – pymatgen Molecule or Structure, Structure is not
  advised since the features will depend on supercell size.
* **Returns**
  pandas.DataFrame.
  The column is index of the structure, which is 0 for single input
  df[0] returns the serials of coulomb_mat raval

## *class* maml.describers.DistinctSiteProperty(properties: list[str], symprec: float = 0.1, wyckoffs: list[str] | None = None, feature_batch: str = ‘pandas_concat’, \*\*kwargs)

Bases: `BaseDescriber`

Constructs a describers based on properties of distinct sites in a
structure. For now, this assumes that there is only one type of species in
a particular Wyckoff site.

Reference:
@article{ye2018deep,

> title={Deep neural networks for accurate predictions of crystal stability},
> author={Ye, Weike and Chen, Chi and Wang, Zhenbin and
> > Chu, Iek-Heng and Ong, Shyue Ping},
> > journal={Nature communications},
> > volume={9},
> > number={1},
> > pages={1–6},
> > year={2018},
> > publisher={Nature Publishing Group}}

### *abc_impl(* = <_abc.*abc_data object* )

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘structure_ )

### supported_properties(_ = [‘mendeleev_no’, ‘electrical_resistivity’, ‘velocity_of_sound’, ‘reflectivity’, ‘refractive_index’, ‘poissons_ratio’, ‘molar_volume’, ‘thermal_conductivity’, ‘boiling_point’, ‘melting_point’, ‘critical_temperature’, ‘superconduction_temperature’, ‘liquid_range’, ‘bulk_modulus’, ‘youngs_modulus’, ‘brinell_hardness’, ‘rigidity_modulus’, ‘mineral_hardness’, ‘vickers_hardness’, ‘density_of_solid’, ‘atomic_radius_calculated’, ‘van_der_waals_radius’, ‘coefficient_of_linear_thermal_expansion’, ‘ground_state_term_symbol’, ‘valence’, ‘Z’, ‘X’, ‘atomic_mass’, ‘block’, ‘row’, ‘group’, ‘atomic_radius’, ‘average_ionic_radius’, ‘average_cationic_radius’, ‘average_anionic_radius’, ‘metallic_radius’, ‘ionic_radii’, ‘oxi_state’, ‘max_oxidation_state’, ‘min_oxidation_state’, ‘is_transition_metal’, ‘is_alkali’, ‘is_alkaline’, ‘is_chalcogen’, ‘is_halogen’, ‘is_lanthanoid’, ‘is_metal’, ‘is_metalloid’, ‘is_noble_gas’, ‘is_post_transition_metal’, ‘is_quadrupolar’, ‘is_rare_earth_metal’, ‘is_actinoid’_ )

### transform_one(structure: Structure)

* **Parameters**
  **structure** (*pymatgen Structure*) – pymatgen structure for descriptor computation.
* **Returns**
  pd.DataFrame that contains the distinct position labeled features

## *class* maml.describers.ElementProperty(\*args, \*\*kwargs)

Bases: `BaseDescriber`

Class to calculate elemental property attributes.

To initialize quickly, use the from_preset() method.

Features: Based on the statistics of the data_source chosen, computed
by element stoichiometry. The format generally is:

“{data source} {statistic} {property}”

For example:

“PymetgenData range X”  # Range of electronegativity from Pymatgen data

For a list of all statistics, see the PropertyStats documentation; for a
list of all attributes available for a given data_source, see the
documentation for the data sources (e.g., PymatgenData, MagpieData,
MatscholarElementData, etc.).

* **Parameters**
  * **data_source** (*AbstractData*\* or \**str*) – source from which to retrieve
    element property data (or use str for preset: “pymatgen”,
    “magpie”, or “deml”)
  * **features** (*list*\* of \**strings*) – List of elemental properties to use
    (these must be supported by data_source)
  * **stats** (*list*\* of \**strings*) – a list of weighted statistics to compute to for each
    property (see PropertyStats for available stats)

### *abc_impl(* = <_abc.*abc_data object* )

### *classmethod* \_get_param_names()

Get parameter names for the estimator

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘composition_ )

### *classmethod* from_preset(name: str, \*\*kwargs)

Wrap matminer_wrapper’s from_preset function.

### get_params(deep=False)

Get parameters for this estimator.

* **Parameters**
  **deep** (*bool*\*, \**default=True*) – If True, will return the parameters for this estimator and
  contained subobjects that are estimators.
* **Returns**
  **params** – Parameter names mapped to their values.
* **Return type**
  dict

### transform_one(obj: Any)

Featurize to transform_one.

## *class* maml.describers.ElementStats(element_properties: dict, stats: list[str] | None = None, property_names: list[str] | None = None, feature_batch: str = ‘pandas_concat’, \*\*kwargs)

Bases: `BaseDescriber`

Element statistics. The allowed stats are accessed via ALLOWED_STATS class
attributes. If the stats have multiple parameters, the positional arguments
are separated by ::, e.g., moment::1::None.

### ALLOWED_STATS(_ = [‘max’, ‘min’, ‘range’, ‘mode’, ‘mean_absolute_deviation’, ‘mean_absolute_error’, ‘moment’, ‘mean’, ‘inverse_mean’, ‘average’, ‘std’, ‘skewness’, ‘kurtosis’, ‘geometric_mean’, ‘power_mean’, ‘shifted_geometric_mean’, ‘harmonic_mean’_ )

### AVAILABLE_DATA(_ = [‘megnet_1’, ‘megnet_3’, ‘megnet_l2’, ‘megnet_ion_l2’, ‘megnet_l3’, ‘megnet_ion_l3’, ‘megnet_l4’, ‘megnet_ion_l4’, ‘megnet_l8’, ‘megnet_ion_l8’, ‘megnet_l16’, ‘megnet_ion_l16’, ‘megnet_l32’, ‘megnet_ion_l32’_ )

### *abc_impl(* = <_abc.*abc_data object* )

### *static* \_reduce_dimension(element_properties, property_names, num_dim: int | None = None, reduction_algo: str | None = ‘pca’, reduction_params: dict | None = None)

Reduce the feature dimension by reduction_algo.

* **Parameters**
  * **element_properties** (*dict*) – dictionary of elemental/specie propeprties
  * **property_names** (*list*) – list of property names
  * **num_dim** (*int*) – number of dimension to keep
  * **reduction_algo** (*str*) – algorithm for dimensional reduction, currently support
    pca, kpca
  * **reduction_params** (*dict*) – kwargs for reduction algorithm

Returns: new element_properties and property_names

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘composition_ )

### *classmethod* from_data(data_name: list[str] | str, stats: list[str] | None = None, \*\*kwargs)

ElementalStats from existing data file.

* **Parameters**
  * **data_name** (*str*\* of **list** of \**str*) – data name. Current supported data are
    available from ElementStats.AVAILABLE_DATA
  * **stats** (*list*) – list of stats, use ElementStats.ALLOWED_STATS to
    check available stats
  * **\*\*kwargs** – Passthrough to class init.

Returns: ElementStats instance

### *classmethod* from_file(filename: str, stats: list[str] | None = None, \*\*kwargs)

ElementStats from a json file of element property dictionary.

The keys required are:

> element_properties
> property_names
* **Parameters**
  * **filename** (*str*) – filename
  * **stats** (*list*) – list of stats, check ElementStats.ALLOWED_STATS
    for supported stats. The stats that support additional
    Keyword args, use ‘:’ to separate the args. For example,
    ‘moment:0:None’ will calculate moment stats with order=0,
    and max_order=None.
  * **\*\*kwargs** – Passthrough to class init.

Returns: ElementStats class

### transform_one(obj: Structure | str | Composition)

Transform one object, the object can be string, Compostion or Structure.

* **Parameters**
  **obj** (*str/Composition/Structure*) – object to transform

Returns: pd.DataFrame with property names as column names

## *class* maml.describers.M3GNetStructure(model_path: str | None = None, \*\*kwargs)

Bases: `BaseDescriber`

### *abc_impl(* = <_abc.*abc_data object* )

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### transform_one(structure: Structure | Molecule)

Transform structure/molecule objects into features
:param structure: target object structure or molecule
:type structure: Structure/Molecule

Returns: np.array features.

## *class* maml.describers.MEGNetSite(name: str | object | None = None, level: int | None = None, \*\*kwargs)

Bases: `BaseDescriber`

Use megnet pre-trained models as featurizer to get atomic features.

Reference:
@article{chen2019graph,title={Graph networks as a universal machine

> > learning framework for molecules and crystals},
> > author={Chen, Chi and Ye, Weike and Zuo, Yunxing and
> ```none
> Zheng, Chen and Ong, Shyue Ping},
> ```
> journal={Chemistry of Materials}, volume={31}, number={9},
> pages={3564–3572}, year={2019},publisher={ACS Publications}}

### *abc_impl(* = <_abc.*abc_data object* )

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘site_ )

### transform_one(obj: Structure | Molecule)

Get megnet site features from structure object.

* **Parameters**
  **obj** (*structure*\* or \**molecule*) – pymatgen structure or molecules

Returns:

## *class* maml.describers.MEGNetStructure(name: str | object | None = None, mode: str = ‘site_stats’, level: int | None = None, stats: list | None = None, \*\*kwargs)

Bases: `BaseDescriber`

Use megnet pre-trained models as featurizer to get
structural features. There are two methods to get structural descriptors from
megnet models.

mode:

```none
‘site_stats’: Calculate the site features, and then use maml.utils.stats to compute the feature-wise

    statistics. This requires the specification of level

‘site_readout’: Use the atomic features at the readout stage
‘final’: Use the concatenated atom, bond and global features
```

Reference:
@article{chen2019graph,title={Graph networks as a universal machine

> > learning framework for molecules and crystals},
> > author={Chen, Chi and Ye, Weike and Zuo, Yunxing and
> ```none
> Zheng, Chen and Ong, Shyue Ping},
> ```
> journal={Chemistry of Materials}, volume={31}, number={9},
> pages={3564–3572}, year={2019},publisher={ACS Publications}}

### *abc_impl(* = <_abc.*abc_data object* )

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘structure_ )

### transform_one(obj: Structure | Molecule)

Transform structure/molecule objects into features
:param obj: target object structure or molecule.
:type obj: Structure/Molecule

Returns: pd.DataFrame features

## *class* maml.describers.RadialDistributionFunction(r_min: float = 0.0, r_max: float = 10.0, n_grid: int = 101, sigma: float = 0.0)

Bases: `object`

Calculator for radial distribution function.

### *static* \_get_specie_density(structure: Structure)

### get_site_coordination(structure: Structure)

Get site wise coordination
:param structure: pymatgen Structure.
:type structure: Structure

Returns: r, cns where cns is a list of dictionary with specie_pair: pair_cn key:value pairs

### get_site_rdf(structure: Structure)

* **Parameters**
  **structure** (*Structure*) – pymatgen structure
* **Returns**
  r, rdfs, r is the radial points, and rdfs are a list of rdf dicts
  rdfs[0] is the rdfs of the first site. It is a dictionary of {atom_pair: pair_rdf}
  e.g.,

> {“Sr:O”: [0, 0, 0.1, 0.2, ..]}.

### get_species_coordination(structure: Structure, ref_species: list | None = None, species: list | None = None)

Get specie-wise coordination number
:param structure: target structure
:type structure: Structure
:param ref_species: the reference species.

> The rdfs are calculated with these species at the center
* **Parameters**
  **species** (*list*\* of **species** or \**just single specie str*) – the species that we are interested in.
  The rdfs are calculated on these species.

Returns:

### get_species_rdf(structure: Structure, ref_species: list | None = None, species: list | None = None)

Get specie-wise rdf
:param structure: target structure
:type structure: Structure
:param ref_species: the reference species.

> The rdfs are calculated with these species at the center
* **Parameters**
  **species** (*list*\* of **species** or \**just single specie str*) – the species that we are interested in.
  The rdfs are calculated on these species.

Returns:

## *class* maml.describers.RandomizedCoulombMatrix(random_seed: int | None = None, is_ravel: bool = True, \*\*kwargs)

Bases: `CoulombMatrix`

Randomized CoulombMatrix.

Reference:
@article{montavon2013machine,

> title={Machine learning of molecular electronic properties
> ```none
> in chemical compound space},
> ```
> author={Montavon, Gr{‘e}goire and Rupp, Matthias and Gobre,
> ```none
> Vivekanand and Vazquez-Mayagoitia, Alvaro and Hansen, Katja
> and Tkatchenko, Alexandre and M{"u}ller, Klaus-Robert and
> Von Lilienfeld, O Anatole},
> ```
> journal={New Journal of Physics},
> volume={15}, number={9},pages={095003},
> year={2013},publisher={IOP Publishing}}

### *abc_impl(* = <_abc.*abc_data object* )

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘structure_ )

### get_randomized_coulomb_mat(s: Molecule | Structure)

Returns the randomized matrix
(i) take an arbitrary valid Coulomb matrix C
(ii) compute the norm of each row of this Coulomb matrix: row_norms
(iii) draw a zero-mean unit-variance noise vector ε of the same

> size as row_norms.
1. permute the rows and columns of C with the same permutation

> that sorts row_norms + ε.
* **Parameters**
  **s** (*Molecule/Structure*) – pymatgen Molecule or Structure, Structure is not
  advised since the features will depend on supercell size
* **Returns**
  pd.DataFrame randomized Coulomb matrix

### transform_one(s: Molecule | Structure)

Transform one structure to descriptors
:param s: pymatgen Molecule or Structure, Structure is not

> advised since the features will depend on supercell size.

Returns: pandas dataframe descriptors

## *class* maml.describers.SiteElementProperty(feature_dict: dict | None = None, output_weights: bool = False, \*\*kwargs)

Bases: `BaseDescriber`

Site specie property describers. For a structure or composition, return
an unordered set of site specie properties.

### *abc_impl(* = <_abc.*abc_data object* )

### *static* \_get_keys(c: Composition)

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘site_ )

### *property* feature_dim()

Feature dimension.

### transform_one(obj: str | Composition | Structure | Molecule)

Transform one object to features.

* **Parameters**
  **obj** (*str/Composition/Structure/Molecule*) – object to transform
* **Returns**
  features array

## *class* maml.describers.SmoothOverlapAtomicPosition(cutoff: float, l_max: int = 8, n_max: int = 8, atom_sigma: float = 0.5, feature_batch: str = ‘pandas_concat’, \*\*kwargs)

Bases: `BaseDescriber`

Smooth overlap of atomic positions (SOAP) to describe the local environment
of each atom.

Reference:
@article{bartok2013representing,

> title={On representing chemical environments},
> author={Bart{‘o}k, Albert P and Kondor, Risi and Cs{‘a}nyi, G{‘a}bor},
> journal={Physical Review B},
> volume={87}, number={18}, pages={184115}, year={2013}, publisher={APS}}

### *abc_impl(* = <_abc.*abc_data object* )

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘site_ )

### transform_one(structure: Structure)

* **Parameters**
  **structure** (*Structure*) – Pymatgen Structure object.

## *class* maml.describers.SortedCoulombMatrix(random_seed: int | None = None, is_ravel: bool = True, \*\*kwargs)

Bases: `CoulombMatrix`

Sorted CoulombMatrix.

Reference:
@inproceedings{montavon2012learning,

> title={Learning invariant representations
> ```none
> of molecules for atomization energy prediction},
> ```
> author={Montavon, Gr{‘e}goire and Hansen, Katja
> ```none
> and Fazli, Siamac and Rupp, Matthias and Biegler,
> Franziska and Ziehe, Andreas and Tkatchenko, Alexandre
> and Lilienfeld, Anatole V and M{"u}ller, Klaus-Robert},
> ```
> booktitle={Advances in neural information processing systems},
> pages={440–448}, year={2012}}

### *abc_impl(* = <_abc.*abc_data object* )

### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

### describer_type(_ = ‘structure_ )

### get_sorted_coulomb_mat(s: Molecule | Structure)

Returns the matrix sorted by the row norm.

* **Parameters**
  **s** (*Molecule/Structure*) – pymatgen Molecule or Structure, Structure is not
  advised since the features will depend on supercell size
* **Returns**
  pd.DataFrame, sorted Coulomb matrix

### transform_one(s: Molecule | Structure)

Transform one structure into descriptor
:param s: pymatgen Molecule or Structure, Structure is not

> advised since the features will depend on supercell size.

Returns: pd.DataFrame descriptors

## maml.describers.wrap_matminer_describer(cls_name: str, wrapped_class: Any, obj_conversion: Callable, describer_type: Any | None = None)

Wrapper of matminer_wrapper describers.

* **Parameters**
  * **cls_name** (*str*) – new class name
  * **wrapped_class** (*class object*) – matminer_wrapper BaseFeaturizer
  * **obj_conversion** (*callable*) – function to convert objects into desired
    object type within transform_one
  * **describer_type** (*object*) – object type.

Returns: maml describers class

## maml.describers._composition module

Compositional describers.

### *class* maml.describers._composition.ElementStats(element_properties: dict, stats: list[str] | None = None, property_names: list[str] | None = None, feature_batch: str = ‘pandas_concat’, \*\*kwargs)

Bases: `BaseDescriber`

Element statistics. The allowed stats are accessed via ALLOWED_STATS class
attributes. If the stats have multiple parameters, the positional arguments
are separated by ::, e.g., moment::1::None.

#### ALLOWED_STATS(_ = [‘max’, ‘min’, ‘range’, ‘mode’, ‘mean_absolute_deviation’, ‘mean_absolute_error’, ‘moment’, ‘mean’, ‘inverse_mean’, ‘average’, ‘std’, ‘skewness’, ‘kurtosis’, ‘geometric_mean’, ‘power_mean’, ‘shifted_geometric_mean’, ‘harmonic_mean’_ )

#### AVAILABLE_DATA(_ = [‘megnet_1’, ‘megnet_3’, ‘megnet_l2’, ‘megnet_ion_l2’, ‘megnet_l3’, ‘megnet_ion_l3’, ‘megnet_l4’, ‘megnet_ion_l4’, ‘megnet_l8’, ‘megnet_ion_l8’, ‘megnet_l16’, ‘megnet_ion_l16’, ‘megnet_l32’, ‘megnet_ion_l32’_ )

#### *abc_impl(* = <_abc.*abc_data object* )

#### *static* \_reduce_dimension(element_properties, property_names, num_dim: int | None = None, reduction_algo: str | None = ‘pca’, reduction_params: dict | None = None)

Reduce the feature dimension by reduction_algo.

* **Parameters**
  * **element_properties** (*dict*) – dictionary of elemental/specie propeprties
  * **property_names** (*list*) – list of property names
  * **num_dim** (*int*) – number of dimension to keep
  * **reduction_algo** (*str*) – algorithm for dimensional reduction, currently support
    pca, kpca
  * **reduction_params** (*dict*) – kwargs for reduction algorithm

Returns: new element_properties and property_names

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘composition_ )

#### *classmethod* from_data(data_name: list[str] | str, stats: list[str] | None = None, \*\*kwargs)

ElementalStats from existing data file.

* **Parameters**
  * **data_name** (*str*\* of **list** of \**str*) – data name. Current supported data are
    available from ElementStats.AVAILABLE_DATA
  * **stats** (*list*) – list of stats, use ElementStats.ALLOWED_STATS to
    check available stats
  * **\*\*kwargs** – Passthrough to class init.

Returns: ElementStats instance

#### *classmethod* from_file(filename: str, stats: list[str] | None = None, \*\*kwargs)

ElementStats from a json file of element property dictionary.

The keys required are:

> element_properties
> property_names
* **Parameters**
  * **filename** (*str*) – filename
  * **stats** (*list*) – list of stats, check ElementStats.ALLOWED_STATS
    for supported stats. The stats that support additional
    Keyword args, use ‘:’ to separate the args. For example,
    ‘moment:0:None’ will calculate moment stats with order=0,
    and max_order=None.
  * **\*\*kwargs** – Passthrough to class init.

Returns: ElementStats class

#### transform_one(obj: Structure | str | Composition)

Transform one object, the object can be string, Compostion or Structure.

* **Parameters**
  **obj** (*str/Composition/Structure*) – object to transform

Returns: pd.DataFrame with property names as column names

### maml.describers._composition._is_element_or_specie(s: str)

### maml.describers._composition._keys_are_elements(dic: dict)

## maml.describers._m3gnet module

### *class* maml.describers._m3gnet.M3GNetStructure(model_path: str | None = None, \*\*kwargs)

Bases: `BaseDescriber`

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### transform_one(structure: Structure | Molecule)

Transform structure/molecule objects into features
:param structure: target object structure or molecule
:type structure: Structure/Molecule

Returns: np.array features.

## maml.describers._matminer module

Wrapper for matminer_wrapper featurizers.

### maml.describers._matminer.wrap_matminer_describer(cls_name: str, wrapped_class: Any, obj_conversion: Callable, describer_type: Any | None = None)

Wrapper of matminer_wrapper describers.

* **Parameters**
  * **cls_name** (*str*) – new class name
  * **wrapped_class** (*class object*) – matminer_wrapper BaseFeaturizer
  * **obj_conversion** (*callable*) – function to convert objects into desired
    object type within transform_one
  * **describer_type** (*object*) – object type.

Returns: maml describers class

## maml.describers._megnet module

MEGNet-based describers.

### *exception* maml.describers._megnet.MEGNetNotFound()

Bases: `Exception`

MEGNet not found exception.

### *class* maml.describers._megnet.MEGNetSite(name: str | object | None = None, level: int | None = None, \*\*kwargs)

Bases: `BaseDescriber`

Use megnet pre-trained models as featurizer to get atomic features.

Reference:
@article{chen2019graph,title={Graph networks as a universal machine

> > learning framework for molecules and crystals},
> > author={Chen, Chi and Ye, Weike and Zuo, Yunxing and
> ```none
> Zheng, Chen and Ong, Shyue Ping},
> ```
> journal={Chemistry of Materials}, volume={31}, number={9},
> pages={3564–3572}, year={2019},publisher={ACS Publications}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘site_ )

#### transform_one(obj: Structure | Molecule)

Get megnet site features from structure object.

* **Parameters**
  **obj** (*structure*\* or \**molecule*) – pymatgen structure or molecules

Returns:

### *class* maml.describers._megnet.MEGNetStructure(name: str | object | None = None, mode: str = ‘site_stats’, level: int | None = None, stats: list | None = None, \*\*kwargs)

Bases: `BaseDescriber`

Use megnet pre-trained models as featurizer to get
structural features. There are two methods to get structural descriptors from
megnet models.

mode:

```none
‘site_stats’: Calculate the site features, and then use maml.utils.stats to compute the feature-wise

    statistics. This requires the specification of level

‘site_readout’: Use the atomic features at the readout stage
‘final’: Use the concatenated atom, bond and global features
```

Reference:
@article{chen2019graph,title={Graph networks as a universal machine

> > learning framework for molecules and crystals},
> > author={Chen, Chi and Ye, Weike and Zuo, Yunxing and
> ```none
> Zheng, Chen and Ong, Shyue Ping},
> ```
> journal={Chemistry of Materials}, volume={31}, number={9},
> pages={3564–3572}, year={2019},publisher={ACS Publications}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘structure_ )

#### transform_one(obj: Structure | Molecule)

Transform structure/molecule objects into features
:param obj: target object structure or molecule.
:type obj: Structure/Molecule

Returns: pd.DataFrame features

### maml.describers._megnet._load_model(name: str | object | None = None)

## maml.describers._rdf module

Radial distribution functions for site features.
This was originally written in pymatgen-diffusion.

### *class* maml.describers._rdf.RadialDistributionFunction(r_min: float = 0.0, r_max: float = 10.0, n_grid: int = 101, sigma: float = 0.0)

Bases: `object`

Calculator for radial distribution function.

#### *static* \_get_specie_density(structure: Structure)

#### get_site_coordination(structure: Structure)

Get site wise coordination
:param structure: pymatgen Structure.
:type structure: Structure

Returns: r, cns where cns is a list of dictionary with specie_pair: pair_cn key:value pairs

#### get_site_rdf(structure: Structure)

* **Parameters**
  **structure** (*Structure*) – pymatgen structure
* **Returns**
  r, rdfs, r is the radial points, and rdfs are a list of rdf dicts
  rdfs[0] is the rdfs of the first site. It is a dictionary of {atom_pair: pair_rdf}
  e.g.,

> {“Sr:O”: [0, 0, 0.1, 0.2, ..]}.

#### get_species_coordination(structure: Structure, ref_species: list | None = None, species: list | None = None)

Get specie-wise coordination number
:param structure: target structure
:type structure: Structure
:param ref_species: the reference species.

> The rdfs are calculated with these species at the center
* **Parameters**
  **species** (*list*\* of **species** or \**just single specie str*) – the species that we are interested in.
  The rdfs are calculated on these species.

Returns:

#### get_species_rdf(structure: Structure, ref_species: list | None = None, species: list | None = None)

Get specie-wise rdf
:param structure: target structure
:type structure: Structure
:param ref_species: the reference species.

> The rdfs are calculated with these species at the center
* **Parameters**
  **species** (*list*\* of **species** or \**just single specie str*) – the species that we are interested in.
  The rdfs are calculated on these species.

Returns:

### maml.describers._rdf._dist_to_counts(d: ndarray, r_min: float = 0.0, r_max: float = 8.0, n_grid: int = 100)

Convert a distance array for counts in the bin
:param d: distance array
:type d: 1D np.ndarray
:param r_min: minimal radius
:type r_min: float
:param r_max: maximum radius
:type r_max: float

* **Returns**
  1D array of counts in the bins centered on grid.

### maml.describers._rdf.get_pair_distances(structure: Structure, r_max: float = 8.0)

Get pair distances from structure.
The output will be a list of of dictionary, for example
[{“specie”: “Mo”,
“neighbors”: {“S”: [1.0, 2.0, …], “Fe”: [1.2, 3.0, …]}},
{“specie”: “Fe”,
“neighbors”: {“Mo”: [1.0, 3.0, …]}}]
it will be fairly easy to construct radial distribution func, etc
from here.

* **Parameters**
  * **structure** (*Structure*) – pymatgen Structure
  * **r_max** (*float*) – maximum radius to consider

Returns:

## maml.describers._site module

This module provides local environment describers.

### *class* maml.describers._site.BPSymmetryFunctions(cutoff: float, r_etas: ndarray, r_shift: ndarray, a_etas: ndarray, zetas: ndarray, lambdas: ndarray, feature_batch: str = ‘pandas_concat’, \*\*kwargs)

Bases: `BaseDescriber`

Behler-Parrinello symmetry function to describe the local environment
of each atom.

Reference:
@article{behler2007generalized,

> title={Generalized neural-network representation of
> ```none
> high-dimensional potential-energy surfaces},
> ```
> author={Behler, J{“o}rg and Parrinello, Michele},
> journal={Physical review letters},
> volume={98},
> number={14},
> pages={146401},
> year={2007},
> publisher={APS}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### \_fc(r: float)

Cutoff function to decay the symmetry functions at vicinity of radial cutoff.

* **Parameters**
  **r** (*float*) – The pair distance.

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘site_ )

#### transform_one(structure: Structure)

* **Parameters**
  **structure** (*Structure*) – Pymatgen Structure object.

### *class* maml.describers._site.BispectrumCoefficients(rcutfac: float, twojmax: int, element_profile: dict, quadratic: bool = False, pot_fit: bool = False, include_stress: bool = False, feature_batch: str = ‘pandas_concat’, \*\*kwargs)

Bases: `BaseDescriber`

Bispectrum coefficients to describe the local environment of each atom.
Lammps is required to perform this computation.

Reference:
@article{bartok2010gaussian,

> title={Gaussian approximation potentials: The
> ```none
> accuracy of quantum mechanics, without the electrons},
> ```
> author={Bart{‘o}k, Albert P and Payne, Mike C
> ```none
> and Kondor, Risi and Cs{'a}nyi, G{'a}bor},
> ```
> journal={Physical review letters},
> volume={104}, number={13}, pages={136403}, year={2010}, publisher={APS}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘site_ )

#### *property* feature_dim(    *: int | Non* )

Bispectrum feature dimension.

#### *property* subscripts(    *: lis* )

The subscripts (2j1, 2j2, 2j) of all bispectrum components
involved.

#### transform_one(structure: Structure)

* **Parameters**
  **structure** (*Structure*) – Pymatgen Structure object.

### *class* maml.describers._site.MEGNetSite(name: str | object | None = None, level: int | None = None, \*\*kwargs)

Bases: `BaseDescriber`

Use megnet pre-trained models as featurizer to get atomic features.

Reference:
@article{chen2019graph,title={Graph networks as a universal machine

> > learning framework for molecules and crystals},
> > author={Chen, Chi and Ye, Weike and Zuo, Yunxing and
> ```none
> Zheng, Chen and Ong, Shyue Ping},
> ```
> journal={Chemistry of Materials}, volume={31}, number={9},
> pages={3564–3572}, year={2019},publisher={ACS Publications}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘site_ )

#### transform_one(obj: Structure | Molecule)

Get megnet site features from structure object.

* **Parameters**
  **obj** (*structure*\* or \**molecule*) – pymatgen structure or molecules

Returns:

### *class* maml.describers._site.SiteElementProperty(feature_dict: dict | None = None, output_weights: bool = False, \*\*kwargs)

Bases: `BaseDescriber`

Site specie property describers. For a structure or composition, return
an unordered set of site specie properties.

#### *abc_impl(* = <_abc.*abc_data object* )

#### *static* \_get_keys(c: Composition)

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘site_ )

#### *property* feature_dim()

Feature dimension.

#### transform_one(obj: str | Composition | Structure | Molecule)

Transform one object to features.

* **Parameters**
  **obj** (*str/Composition/Structure/Molecule*) – object to transform
* **Returns**
  features array

### *class* maml.describers._site.SmoothOverlapAtomicPosition(cutoff: float, l_max: int = 8, n_max: int = 8, atom_sigma: float = 0.5, feature_batch: str = ‘pandas_concat’, \*\*kwargs)

Bases: `BaseDescriber`

Smooth overlap of atomic positions (SOAP) to describe the local environment
of each atom.

Reference:
@article{bartok2013representing,

> title={On representing chemical environments},
> author={Bart{‘o}k, Albert P and Kondor, Risi and Cs{‘a}nyi, G{‘a}bor},
> journal={Physical Review B},
> volume={87}, number={18}, pages={184115}, year={2013}, publisher={APS}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘site_ )

#### transform_one(structure: Structure)

* **Parameters**
  **structure** (*Structure*) – Pymatgen Structure object.

## maml.describers._spectrum module

Spectrum describers.

## maml.describers._structure module

Structure-wise describers. These describers include structural information.

### *class* maml.describers._structure.CoulombEigenSpectrum(max_atoms: int | None = None, \*\*kwargs)

Bases: `BaseDescriber`

Get the Coulomb Eigen Spectrum describers.

Reference:
@article{rupp2012fast,

> title={Fast and accurate modeling of molecular
> ```none
> atomization energies with machine learning},
> ```
> author={Rupp, Matthias and Tkatchenko, Alexandre and M{“u}ller,
> ```none
> Klaus-Robert and Von Lilienfeld, O Anatole},
> ```
> journal={Physical review letters}, volume={108},
> number={5}, pages={058301},
> year={2012}, publisher={APS}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘structure_ )

#### transform_one(mol: Molecule)

* **Parameters**
  **mol** (*Molecule*) – pymatgen molecule.

Returns: np.ndarray the eigen value vectors of Coulob matrix

### *class* maml.describers._structure.CoulombMatrix(random_seed: int | None = None, max_atoms: int | None = None, is_ravel: bool = True, \*\*kwargs)

Bases: `BaseDescriber`

Coulomb Matrix to describe structure.

Reference:
@article{rupp2012fast,

> title={Fast and accurate modeling of molecular
> ```none
> atomization energies with machine learning},
> ```
> author={Rupp, Matthias and Tkatchenko, Alexandre and M{“u}ller,
> ```none
> Klaus-Robert and Von Lilienfeld, O Anatole},
> ```
> journal={Physical review letters}, volume={108},
> number={5}, pages={058301},
> year={2012}, publisher={APS}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### *static* \_get_columb_mat(s: Molecule | Structure)

* **Parameters**
  **s** (*Molecule/Structure*) – input Molecule or Structure. Structure
  is not advised since the feature will depend on the supercell size.
* **Returns**
  Coulomb matrix of the structure

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘structure_ )

#### get_coulomb_mat(s: Molecule | Structure)

* **Parameters**
  **s** (*Molecule/Structure*) – input Molecule or Structure. Structure
  is not advised since the feature will depend on the supercell size
* **Returns**
  Coulomb matrix of the structure.

#### transform_one(s: Molecule | Structure)

* **Parameters**
  **s** (*Molecule/Structure*) – pymatgen Molecule or Structure, Structure is not
  advised since the features will depend on supercell size.
* **Returns**
  pandas.DataFrame.
  The column is index of the structure, which is 0 for single input
  df[0] returns the serials of coulomb_mat raval

### *class* maml.describers._structure.DistinctSiteProperty(properties: list[str], symprec: float = 0.1, wyckoffs: list[str] | None = None, feature_batch: str = ‘pandas_concat’, \*\*kwargs)

Bases: `BaseDescriber`

Constructs a describers based on properties of distinct sites in a
structure. For now, this assumes that there is only one type of species in
a particular Wyckoff site.

Reference:
@article{ye2018deep,

> title={Deep neural networks for accurate predictions of crystal stability},
> author={Ye, Weike and Chen, Chi and Wang, Zhenbin and
> > Chu, Iek-Heng and Ong, Shyue Ping},
> > journal={Nature communications},
> > volume={9},
> > number={1},
> > pages={1–6},
> > year={2018},
> > publisher={Nature Publishing Group}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘structure_ )

#### supported_properties(_ = [‘mendeleev_no’, ‘electrical_resistivity’, ‘velocity_of_sound’, ‘reflectivity’, ‘refractive_index’, ‘poissons_ratio’, ‘molar_volume’, ‘thermal_conductivity’, ‘boiling_point’, ‘melting_point’, ‘critical_temperature’, ‘superconduction_temperature’, ‘liquid_range’, ‘bulk_modulus’, ‘youngs_modulus’, ‘brinell_hardness’, ‘rigidity_modulus’, ‘mineral_hardness’, ‘vickers_hardness’, ‘density_of_solid’, ‘atomic_radius_calculated’, ‘van_der_waals_radius’, ‘coefficient_of_linear_thermal_expansion’, ‘ground_state_term_symbol’, ‘valence’, ‘Z’, ‘X’, ‘atomic_mass’, ‘block’, ‘row’, ‘group’, ‘atomic_radius’, ‘average_ionic_radius’, ‘average_cationic_radius’, ‘average_anionic_radius’, ‘metallic_radius’, ‘ionic_radii’, ‘oxi_state’, ‘max_oxidation_state’, ‘min_oxidation_state’, ‘is_transition_metal’, ‘is_alkali’, ‘is_alkaline’, ‘is_chalcogen’, ‘is_halogen’, ‘is_lanthanoid’, ‘is_metal’, ‘is_metalloid’, ‘is_noble_gas’, ‘is_post_transition_metal’, ‘is_quadrupolar’, ‘is_rare_earth_metal’, ‘is_actinoid’_ )

#### transform_one(structure: Structure)

* **Parameters**
  **structure** (*pymatgen Structure*) – pymatgen structure for descriptor computation.
* **Returns**
  pd.DataFrame that contains the distinct position labeled features

### *class* maml.describers._structure.RandomizedCoulombMatrix(random_seed: int | None = None, is_ravel: bool = True, \*\*kwargs)

Bases: `CoulombMatrix`

Randomized CoulombMatrix.

Reference:
@article{montavon2013machine,

> title={Machine learning of molecular electronic properties
> ```none
> in chemical compound space},
> ```
> author={Montavon, Gr{‘e}goire and Rupp, Matthias and Gobre,
> ```none
> Vivekanand and Vazquez-Mayagoitia, Alvaro and Hansen, Katja
> and Tkatchenko, Alexandre and M{"u}ller, Klaus-Robert and
> Von Lilienfeld, O Anatole},
> ```
> journal={New Journal of Physics},
> volume={15}, number={9},pages={095003},
> year={2013},publisher={IOP Publishing}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘structure_ )

#### get_randomized_coulomb_mat(s: Molecule | Structure)

Returns the randomized matrix
(i) take an arbitrary valid Coulomb matrix C
(ii) compute the norm of each row of this Coulomb matrix: row_norms
(iii) draw a zero-mean unit-variance noise vector ε of the same

> size as row_norms.
1. permute the rows and columns of C with the same permutation

> that sorts row_norms + ε.
* **Parameters**
  **s** (*Molecule/Structure*) – pymatgen Molecule or Structure, Structure is not
  advised since the features will depend on supercell size
* **Returns**
  pd.DataFrame randomized Coulomb matrix

#### transform_one(s: Molecule | Structure)

Transform one structure to descriptors
:param s: pymatgen Molecule or Structure, Structure is not

> advised since the features will depend on supercell size.

Returns: pandas dataframe descriptors

### *class* maml.describers._structure.SortedCoulombMatrix(random_seed: int | None = None, is_ravel: bool = True, \*\*kwargs)

Bases: `CoulombMatrix`

Sorted CoulombMatrix.

Reference:
@inproceedings{montavon2012learning,

> title={Learning invariant representations
> ```none
> of molecules for atomization energy prediction},
> ```
> author={Montavon, Gr{‘e}goire and Hansen, Katja
> ```none
> and Fazli, Siamac and Rupp, Matthias and Biegler,
> Franziska and Ziehe, Andreas and Tkatchenko, Alexandre
> and Lilienfeld, Anatole V and M{"u}ller, Klaus-Robert},
> ```
> booktitle={Advances in neural information processing systems},
> pages={440–448}, year={2012}}

#### *abc_impl(* = <_abc.*abc_data object* )

#### *sklearn_auto_wrap_output_keys(* = {‘transform’_ )

#### describer_type(_ = ‘structure_ )

#### get_sorted_coulomb_mat(s: Molecule | Structure)

Returns the matrix sorted by the row norm.

* **Parameters**
  **s** (*Molecule/Structure*) – pymatgen Molecule or Structure, Structure is not
  advised since the features will depend on supercell size
* **Returns**
  pd.DataFrame, sorted Coulomb matrix

#### transform_one(s: Molecule | Structure)

Transform one structure into descriptor
:param s: pymatgen Molecule or Structure, Structure is not

> advised since the features will depend on supercell size.

Returns: pd.DataFrame descriptors