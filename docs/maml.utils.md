---
layout: default
title: maml.utils.md
nav_exclude: true
---

# maml.utils package

Utilities package.

## *class* maml.utils.ConstantValue(value: float, \*\*kwargs)

Bases: `ValueProfile`

Return constant value.

### get_value()

Return constant value.

## *class* maml.utils.DataSplitter()

Bases: `MSONable`

Data splitter base class.

### split(mat_ids, \*\*kwargs)

Split the mat_ids, optionally one can provide
targets. This is useful in stratified split.

* **Parameters**
  **mat_ids** (*list*) – list of material ids

Returns: (train_ids, val_ids, test_ids) or

```none
(train_ids, test_ids)
```

## *class* maml.utils.DummyScaler()

Bases: `MSONable`

Dummy scaler does nothing.

### *classmethod* from_training_data(structures: list[StructureOrMolecule], targets: VectorLike, is_intensive: bool = True)

* **Parameters**
  * **structures** (*list*) – list of structures/molecules
  * **targets** (*list*) – vector of target properties
  * **is_intensive** (*bool*) – whether the target is intensive

Returns: DummyScaler.

### *static* inverse_transform(transformed_target: float, n: int = 1)

return as it is
:param transformed_target: transformed target
:type transformed_target: float
:param n: number of atoms
:type n: int

* **Returns**
  transformed_target.

### *static* transform(target: float, n: int = 1)

* **Parameters**
  * **target** (*float*) – target numerical value
  * **n** (*int*) – number of atoms
* **Returns**
  target.

## *class* maml.utils.LinearProfile(value_start: float, value_end: float = 0.0, max_steps: int = 100, \*\*kwargs)

Bases: `ValueProfile`

LinearProfile by setting starting value and the rate of
value change. The profile can be initialized either by
[value_start, value_end, max_step] or [value_start, rate].

### get_value()

Get LinearProfile value
Returns: float.

## *class* maml.utils.MultiScratchDir(rootpath: str | Path, n_dirs: int = 1, create_symbolic_link: bool = False, copy_from_current_on_enter: bool = False, copy_to_current_on_exit: bool = False)

Bases: `object`

Creates a “with” context manager that automatically handles creation of
temporary directories (utilizing Python’s build in temp directory
functions) and cleanup when done. The main difference between this class
and monty ScratchDir is that multiple temp directories are created here.
It enables the running of multiple jobs simultaneously in the directories
The way it works is as follows:

1. Create multiple temp dirs in specified root path.
2. Optionally copy input files from current directory to temp dir.
3. User loops among all directories
4. User performs specified operations in each directories
5. Change back to original directory.
6. Delete temp dir.

### SCR_LINK(_ = ‘scratch_link_ )

## *class* maml.utils.Scaler()

Bases: `MSONable`

Base Scaler class. It implements transform and
inverse_transform. Both methods will take number
of atom as the second parameter in addition to
the target property.

### inverse_transform(transformed_target: float, n: int = 1)

Inverse transform of the target
:param transformed_target: transformed target
:type transformed_target: float
:param n: number of atoms
:type n: int

* **Returns**
  target.

### transform(target: float, n: int = 1)

Transform the target values into new target values
:param target: target numerical value
:type target: float
:param n: number of atoms
:type n: int

* **Returns**
  scaled target.

## *class* maml.utils.ShuffleSplitter(ratios: str = ‘80/10/10’, delim: str = ‘/’, random_seed: int | None = None)

Bases: `DataSplitter`

Randomly shuffe the material ids and split the ids
into given ratios.

### split(mat_ids, \*\*kwargs)

Randomly split the mat_ids
:param mat_ids: material ids
:type mat_ids: list

Returns:

## *class* maml.utils.StandardScaler(mean: float = 0.0, std: float = 1.0, is_intensive: bool = True)

Bases: `Scaler`

Standard scaler with consideration of extensive/intensive quantity
For intensive quantity, the mean is just the mean of training data,
and std is the std of training data
For extensive quantity, the mean is the mean of target/atom, and
std is the std for target/atom
.. method:: transform(self, target, n=1)

> standard scaling the target and.

### *classmethod* from_training_data(structures: list[StructureOrMolecule], targets: VectorLike, is_intensive: bool = True)

Generate a target scaler from a list of input structures/molecules,
a target value vector and an indicator for intensiveness of the
property
:param structures: list of structures/molecules
:type structures: list
:param targets: vector of target properties
:type targets: list
:param is_intensive: whether the target is intensive
:type is_intensive: bool

Returns: new instance.

### inverse_transform(transformed_target: float, n: int = 1)

Inverse transform of the target
:param transformed_target: transformed target
:type transformed_target: float
:param n: number of atoms
:type n: int

* **Returns**
  original target.

### transform(target: float, n: int = 1)

Transform numeric values according the mean and std, plus a factor n
:param target: target numerical value
:type target: float
:param n: number of atoms
:type n: int

* **Returns**
  scaled target.

## *class* maml.utils.Stats()

Bases: `object`

Calculate the stats of a list of values.
This is particularly helpful when you want to convert
lists of values of different lengths to uniform length
for machine learning purposes.

supported

### allowed_stats(_ = [‘max’, ‘min’, ‘range’, ‘mode’, ‘mean_absolute_deviation’, ‘mean_absolute_error’, ‘moment’, ‘mean’, ‘inverse_mean’, ‘average’, ‘std’, ‘skewness’, ‘kurtosis’, ‘geometric_mean’, ‘power_mean’, ‘shifted_geometric_mean’, ‘harmonic_mean’_ )

### *static* average(data: list[float], weights: list[float] | None = None)

Weighted average.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: average value

### *static* geometric_mean(data: list[float], weights: list[float] | None = None)

Geometric mean of the data.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: geometric mean of the distribution

### *static* harmonic_mean(data: list[float], weights: list[float] | None = None)

harmonic mean of the data.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: harmonic mean of the distribution

### *static* inverse_mean(data: list[float], weights: list[float] | None = None)

inverse mean.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: average value

### *static* kurtosis(data: list[float], weights: list[float] | None = None)

Kurtosis of the distribution.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: Kurtosis of the distribution

### *static* max(data: list[float], weights: list[float] | None = None)

Max of value
Args:31

> data (list): list of float data
> weights (list): optional weights.

Returns: maximum value

### *static* mean(data: list[float], weights: list[float] | None = None)

Weighted average.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: average value

### *static* mean_absolute_deviation(data: list[float], weights: list[float] | None = None)

mean absolute deviation.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*) – optional weights

Returns: mean absolute deviation

### *static* mean_absolute_error(data: list[float], weights: list[float] | None = None)

mean absolute error.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*) – optional weights

Returns: mean absolute error

### *static* min(data: list[float], weights: list[float] | None = None)

min of value
:param data: list of float data
:type data: list
:param weights: optional weights.
:type weights: list

Returns: minimum value

### *static* mode(data: list[float], weights: list[float] | None = None)

Mode of data, if multiple entries have equal counts,
compute the average of those.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*) – optional weights

Returns: mode of values, i.e., max - min

### *static* moment(data: list[float], weights: list[float] | None = None, order: int | None = None, max_order: int | None = None)

Moment of probability mass function.

order = 1 means weighted mean
order = 2 means standard deviation
order > 2 corresponds to higher order moment to

> the 1./order power
* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data points
  * **order** (*int*) – moment order
  * **max_order** (*int*) – if set, it will overwrite order

Returns: float or list of floats

### *static* power_mean(data: list[float], weights: list[float] | None = None, p: int = 1)

power mean [https://en.wikipedia.org/wiki/Generalized_mean](https://en.wikipedia.org/wiki/Generalized_mean).

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point
  * **p** (*int*) – power

Returns: power mean of the distribution

### *static* range(data: list[float], weights: list[float] | None = None)

Range of values
:param data: list of float data
:type data: list
:param weights: optional weights.
:type weights: list

Returns: range of values, i.e., max - min

### *static* shifted_geometric_mean(data: list[float], weights: list[float] | None = None, shift: float = 100)

Since we cannot calculate the geometric means on negative or zero values,
we can first shift all values to positive and then calculate the geometric mean
afterwards, we shift the computed geometric mean back by a shift value.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point
  * **shift** (*float*) – shift value

Returns: geometric mean of the distribution

### *static* skewness(data: list[float], weights: list[float] | None = None)

Skewness of the distribution.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: Skewness of the distribution

### *static* std(data: list[float], weights: list[float] | None = None)

Standard deviation.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: Standard deviation

## *class* maml.utils.ValueProfile(max_steps: int | None = None, \*\*kwargs)

Bases: `object`

Base class for ValueProfile. The base class has the following methods
.. method:: increment_step(self)

> add one to step

### get_value(self)

abstract method that return the value.

### get_value()

abstract method that returns the current value
Returns: value float.

### increment_step()

Increase step attribute by one.

## maml.utils.check_structures_forces_stresses(structures: list[Structure], forces: list | None = None, stresses: list | None = None, stress_format: str = ‘VASP’, return_none: bool = True)

Check structures, forces and stresses. The forces and stress are dependent
on the lattice orientation. This function will rotate the structures
and the corresponding forces and structures to lammps format
[[ax, 0, 0],
[bx, by, 0],
[cx, cy, cz]].

The lattice are formed by the row vectors.

* **Parameters**
  * **structures** (*list*) – list of structures
  * **forces** (*list*) – list of force matrixs (m, 3)
  * **stresses** (*list*) – list of stress vectors
  * **stress_format** (*str*) – stress format, choose from
    “VASP”, “LAMMPS”, “SNAP”
  * **return_none** (*bool*) – whether to return list of None
    for forces and stresses

Returns: structures [forces], [stresses]

## maml.utils.convert_docs(docs, include_stress=False, \*\*kwargs)

Method to convert a list of docs into objects, e.g.,
Structure and DataFrame.

* **Parameters**
  * **docs** (    *[**dict**]*) – List of docs. Each doc should have the same
    format as one returned from .dft.parse_dir.
  * **include_stress** (*bool*) – Whether to include stress components.
  * **\*\*kwargs** – Passthrough.
* **Returns**
  A list of structures, and a DataFrame with energy and force
  data in ‘y_orig’ column, data type (‘energy’ or ‘force’) in
  ‘dtype’ column, No. of atoms in ‘n’ column sharing the same row
  of energy data while ‘n’ being 1 for the rows of force data.

## maml.utils.cwt(z: np.ndarray, widths: np.ndarray, wavelet: str | Callable = ‘morlet2’, \*\*kwargs)

The scalogram of the signal
:param z: 1D signal array
:type z: np.ndarray
:param widths: wavelet widths
:type widths: np.ndarray
:param wavelet: wavelet name
:type wavelet: str

Returns: 2D scalogram.

## maml.utils.feature_dim_from_test_system(describer)

Get feature size from a test system.

* **Parameters**
  **describer** (*BaseDescriber*) – describers instance

## maml.utils.fft_magnitude(z: ndarray)

Discrete Fourier Transform the signal z and return
the magnitude of the  coefficients
:param z: 1D signal array
:type z: np.ndarray

Returns: 1D magnitude.

## maml.utils.get_describer_dummy_obj(instance)

For a describers, get a dummy object for transform_one.
This relies on the type hint.

* **Parameters**
  **instance** (*BaseDescriber*) – describers instance

## maml.utils.get_full_args(func: Callable)

Get args from function.

* **Parameters**
  **func** (*callable*) – function to determine the args

## maml.utils.get_full_stats_and_funcs(stats: list)

Get expanded stats function name str and the corresponding
function callables.

* **Parameters**
  **stats** (*list*) – a list of stats names, e.g, [‘mean’, ‘std’, ‘moment:1:None’]

Returns: list of stats names, list of stats callable

## maml.utils.get_lammps_lattice_and_rotation(structure: Structure, origin=(0, 0, 0))

Transform structure to lammps compatible structure. The lattice and rotation
matrix are returned.

* **Parameters**
  * **structure** (*Structure*) – pymatgen structure
  * **origin** (*tuple*) – origin coordinates

Returns: new lattice, rotation symmetry operator, rotation matrix

## maml.utils.get_sp_method(sp_method: str | Callable)

Providing a signal processing method name return the callable
:param sp_method: name of the sp function
:type sp_method: str

Returns: callable for signal processing.

## maml.utils.njit(func: Callable)

Dummy decorator, returns the original function
:param func: function to be wrapped.
:type func: Callable

Returns: decorated function

## maml.utils.pool_from(structures, energies=None, forces=None, stresses=None)

Method to convert structures and their properties in to
datapool format.

* **Parameters**
  * **structures** (    *[**Structure**]*) – The list of Pymatgen Structure object.
  * **energies** (    *[**float**]*) – The list of total energies of each structure
    in structures list.
  * **forces** (    *[**np.array**]*) – List of (m, 3) forces array of each structure
    with m atoms in structures list. m can be varied with each
    single structure case.
  * **stresses** (*list*) – List of (6, ) virial stresses of each
    structure in structures list.
* **Returns**
  ([dict])

## maml.utils.spectrogram(z: np.ndarray, return_time_freq: bool = False)

The spectrogram of the signal
:param z: 1D signal array
:type z: np.ndarray
:param return_time_freq: whether to return time and frequency
:type return_time_freq: bool

Returns: 2D spectrogram.

## maml.utils.stats_list_conversion(stats_list: list[str])

Convert a list of stats str into a fully expanded list.
This applies mainly to stats that can return a list of values, e.g.,
moment with max_order > 1.

* **Parameters**
  **stats_list** (*list*) – list of stats str

Returns: list of expanded stats str

## maml.utils.stress_format_change(stress: np.ndarray | list[float], from_format: str, to_format: str)

Convert stress format from from_format to to_format
:param stress: length-6 stress vector
:type stress: list of float
:param from_format: choose from “VASP”, “LAMMPS”, “SNAP”
:type from_format: str
:param to_format: choose from “VASP”, “LAMMPS”, “SNAP”.
:type to_format: str

Returns: list of float stress vector

## maml.utils.stress_list_to_matrix(stress: np.ndarray | list[float], stress_format: str = ‘VASP’)

convert a length-6 stress list to stress matrix 3x3.

* **Parameters**
  * **stress** (*list*\* of \**float*) – list of stress
  * **stress_format** (*str*) – Supported formats are the follows
    VASP: xx, yy, zz, xy, yz, xz
    LAMMPS: xx, yy, zz, xy, zx, yz
    SNAP: xx, yy, zz, yz, xz, xy

Returns: 3x3 stress matrix

## maml.utils.stress_matrix_to_list(stress_matrix: ndarray, stress_format: str = ‘VASP’)

Stress matrix to list representation
:param stress_matrix: stress matrix 3x3
:type stress_matrix: np.ndarray
:param stress_format: stress list format
:type stress_format: str

Returns: list of float stress vector.

## maml.utils.to_array(x)

Convert x into numerical array
:param x: x can be a dataframe, a list or an array

return np.ndarray.

## maml.utils.to_composition(obj: Composition | Molecule | Structure | str)

Convert str/structure or composition to compositions.

* **Parameters**
  **obj** (*str/structure/composition*) – object to convert
* **Returns**
  Composition object

## maml.utils.write_data_from_structure(structure: Structure, filename: str, ff_elements: list[str] | None = None, significant_figures: int = 6, origin: tuple = (0, 0, 0))

Write structure to lammps data file, this is to speed up
pymatgen LammpsData.

Args:a

```none
structure (Structure): pymatgen structure
filename (str): filename
ff_elements (list of str): elements to be considered
significant_figures (int): significant figures of floats in output
origin (tuple): origin coordinates
```

## maml.utils.wvd(z: np.ndarray, return_all: bool = False)

Wigner Ville Distribution calculator
:param z: signal 1D
:type z: np.ndarray
:param return_all: whether to return time and freq info, default

> only return the wvd information

Returns: NxN wvd matrix.

## maml.utils._data_conversion module

Convert data list to docs or pool existing data lists for training.

### maml.utils._data_conversion.convert_docs(docs, include_stress=False, \*\*kwargs)

Method to convert a list of docs into objects, e.g.,
Structure and DataFrame.

* **Parameters**
  * **docs** (    *[**dict**]*) – List of docs. Each doc should have the same
    format as one returned from .dft.parse_dir.
  * **include_stress** (*bool*) – Whether to include stress components.
  * **\*\*kwargs** – Passthrough.
* **Returns**
  A list of structures, and a DataFrame with energy and force
  data in ‘y_orig’ column, data type (‘energy’ or ‘force’) in
  ‘dtype’ column, No. of atoms in ‘n’ column sharing the same row
  of energy data while ‘n’ being 1 for the rows of force data.

### maml.utils._data_conversion.doc_from(structure, energy=None, force=None, stress=None)

Method to convert structure and its properties into doc
format for further processing. If properties are None, zeros
array will be used.

* **Parameters**
  * **structure** (*Structure*) – Pymatgen Structure object.
  * **energy** (*float*) – The total energy of the structure.
  * **force** (*np.array*) – The (m, 3) forces array of the structure
    where m is the number of atoms in structure.
  * **stress** (*list/np.array*) – The (6, ) stresses array of the
    structure.
* **Returns**
  (dict)

### maml.utils._data_conversion.pool_from(structures, energies=None, forces=None, stresses=None)

Method to convert structures and their properties in to
datapool format.

* **Parameters**
  * **structures** (    *[**Structure**]*) – The list of Pymatgen Structure object.
  * **energies** (    *[**float**]*) – The list of total energies of each structure
    in structures list.
  * **forces** (    *[**np.array**]*) – List of (m, 3) forces array of each structure
    with m atoms in structures list. m can be varied with each
    single structure case.
  * **stresses** (*list*) – List of (6, ) virial stresses of each
    structure in structures list.
* **Returns**
  ([dict])

### maml.utils._data_conversion.to_array(x)

Convert x into numerical array
:param x: x can be a dataframe, a list or an array

return np.ndarray.

## maml.utils._data_split module

Data split.

### *class* maml.utils._data_split.DataSplitter()

Bases: `MSONable`

Data splitter base class.

#### split(mat_ids, \*\*kwargs)

Split the mat_ids, optionally one can provide
targets. This is useful in stratified split.

* **Parameters**
  **mat_ids** (*list*) – list of material ids

Returns: (train_ids, val_ids, test_ids) or

```none
(train_ids, test_ids)
```

### *class* maml.utils._data_split.ShuffleSplitter(ratios: str = ‘80/10/10’, delim: str = ‘/’, random_seed: int | None = None)

Bases: `DataSplitter`

Randomly shuffe the material ids and split the ids
into given ratios.

#### split(mat_ids, \*\*kwargs)

Randomly split the mat_ids
:param mat_ids: material ids
:type mat_ids: list

Returns:

## maml.utils._dummy module

Dummy test systems.

### maml.utils._dummy.feature_dim_from_test_system(describer)

Get feature size from a test system.

* **Parameters**
  **describer** (*BaseDescriber*) – describers instance

### maml.utils._dummy.get_describer_dummy_obj(instance)

For a describers, get a dummy object for transform_one.
This relies on the type hint.

* **Parameters**
  **instance** (*BaseDescriber*) – describers instance

## maml.utils._inspect module

Inspect function args.

### maml.utils._inspect.get_full_args(func: Callable)

Get args from function.

* **Parameters**
  **func** (*callable*) – function to determine the args

### maml.utils._inspect.get_param_types(func)

Get param and type info.

* **Parameters**
  **func** (*callable*) – function to determine the arg types

## maml.utils._jit module

Simple numba utility.
Some functions can excelerated substantially with numba.

### maml.utils._jit.njit(func: Callable)

Dummy decorator, returns the original function
:param func: function to be wrapped.
:type func: Callable

Returns: decorated function

## maml.utils._lammps module

LAMMPS utility.

### maml.utils._lammps._get_atomic_mass(element_or_specie: str)

Get atomic mass from element or specie string.

* **Parameters**
  **element_or_specie** (*str*) – specie or element string

Returns: float mass

### maml.utils._lammps._get_charge(element_or_specie: str | Element | Species)

Get charge from element or specie.

* **Parameters**
  **element_or_specie** (*str*\* or **Element** or \**Species*) – element or specie

Returns: charge float

### maml.utils._lammps.check_structures_forces_stresses(structures: list[Structure], forces: list | None = None, stresses: list | None = None, stress_format: str = ‘VASP’, return_none: bool = True)

Check structures, forces and stresses. The forces and stress are dependent
on the lattice orientation. This function will rotate the structures
and the corresponding forces and structures to lammps format
[[ax, 0, 0],
[bx, by, 0],
[cx, cy, cz]].

The lattice are formed by the row vectors.

* **Parameters**
  * **structures** (*list*) – list of structures
  * **forces** (*list*) – list of force matrixs (m, 3)
  * **stresses** (*list*) – list of stress vectors
  * **stress_format** (*str*) – stress format, choose from
    “VASP”, “LAMMPS”, “SNAP”
  * **return_none** (*bool*) – whether to return list of None
    for forces and stresses

Returns: structures [forces], [stresses]

### maml.utils._lammps.get_lammps_lattice_and_rotation(structure: Structure, origin=(0, 0, 0))

Transform structure to lammps compatible structure. The lattice and rotation
matrix are returned.

* **Parameters**
  * **structure** (*Structure*) – pymatgen structure
  * **origin** (*tuple*) – origin coordinates

Returns: new lattice, rotation symmetry operator, rotation matrix

### maml.utils._lammps.stress_format_change(stress: np.ndarray | list[float], from_format: str, to_format: str)

Convert stress format from from_format to to_format
:param stress: length-6 stress vector
:type stress: list of float
:param from_format: choose from “VASP”, “LAMMPS”, “SNAP”
:type from_format: str
:param to_format: choose from “VASP”, “LAMMPS”, “SNAP”.
:type to_format: str

Returns: list of float stress vector

### maml.utils._lammps.stress_list_to_matrix(stress: np.ndarray | list[float], stress_format: str = ‘VASP’)

convert a length-6 stress list to stress matrix 3x3.

* **Parameters**
  * **stress** (*list*\* of \**float*) – list of stress
  * **stress_format** (*str*) – Supported formats are the follows
    VASP: xx, yy, zz, xy, yz, xz
    LAMMPS: xx, yy, zz, xy, zx, yz
    SNAP: xx, yy, zz, yz, xz, xy

Returns: 3x3 stress matrix

### maml.utils._lammps.stress_matrix_to_list(stress_matrix: ndarray, stress_format: str = ‘VASP’)

Stress matrix to list representation
:param stress_matrix: stress matrix 3x3
:type stress_matrix: np.ndarray
:param stress_format: stress list format
:type stress_format: str

Returns: list of float stress vector.

### maml.utils._lammps.write_data_from_structure(structure: Structure, filename: str, ff_elements: list[str] | None = None, significant_figures: int = 6, origin: tuple = (0, 0, 0))

Write structure to lammps data file, this is to speed up
pymatgen LammpsData.

Args:a

```none
structure (Structure): pymatgen structure
filename (str): filename
ff_elements (list of str): elements to be considered
significant_figures (int): significant figures of floats in output
origin (tuple): origin coordinates
```

## maml.utils._material module

Materials utils.

### maml.utils._material.to_composition(obj: Composition | Molecule | Structure | str)

Convert str/structure or composition to compositions.

* **Parameters**
  **obj** (*str/structure/composition*) – object to convert
* **Returns**
  Composition object

## maml.utils._preprocessing module

Target preprocessing.

### *class* maml.utils._preprocessing.DummyScaler()

Bases: `MSONable`

Dummy scaler does nothing.

#### *classmethod* from_training_data(structures: list[StructureOrMolecule], targets: VectorLike, is_intensive: bool = True)

* **Parameters**
  * **structures** (*list*) – list of structures/molecules
  * **targets** (*list*) – vector of target properties
  * **is_intensive** (*bool*) – whether the target is intensive

Returns: DummyScaler.

#### *static* inverse_transform(transformed_target: float, n: int = 1)

return as it is
:param transformed_target: transformed target
:type transformed_target: float
:param n: number of atoms
:type n: int

* **Returns**
  transformed_target.

#### *static* transform(target: float, n: int = 1)

* **Parameters**
  * **target** (*float*) – target numerical value
  * **n** (*int*) – number of atoms
* **Returns**
  target.

### *class* maml.utils._preprocessing.Scaler()

Bases: `MSONable`

Base Scaler class. It implements transform and
inverse_transform. Both methods will take number
of atom as the second parameter in addition to
the target property.

#### inverse_transform(transformed_target: float, n: int = 1)

Inverse transform of the target
:param transformed_target: transformed target
:type transformed_target: float
:param n: number of atoms
:type n: int

* **Returns**
  target.

#### transform(target: float, n: int = 1)

Transform the target values into new target values
:param target: target numerical value
:type target: float
:param n: number of atoms
:type n: int

* **Returns**
  scaled target.

### *class* maml.utils._preprocessing.StandardScaler(mean: float = 0.0, std: float = 1.0, is_intensive: bool = True)

Bases: `Scaler`

Standard scaler with consideration of extensive/intensive quantity
For intensive quantity, the mean is just the mean of training data,
and std is the std of training data
For extensive quantity, the mean is the mean of target/atom, and
std is the std for target/atom
.. method:: transform(self, target, n=1)

> standard scaling the target and.

#### *classmethod* from_training_data(structures: list[StructureOrMolecule], targets: VectorLike, is_intensive: bool = True)

Generate a target scaler from a list of input structures/molecules,
a target value vector and an indicator for intensiveness of the
property
:param structures: list of structures/molecules
:type structures: list
:param targets: vector of target properties
:type targets: list
:param is_intensive: whether the target is intensive
:type is_intensive: bool

Returns: new instance.

#### inverse_transform(transformed_target: float, n: int = 1)

Inverse transform of the target
:param transformed_target: transformed target
:type transformed_target: float
:param n: number of atoms
:type n: int

* **Returns**
  original target.

#### transform(target: float, n: int = 1)

Transform numeric values according the mean and std, plus a factor n
:param target: target numerical value
:type target: float
:param n: number of atoms
:type n: int

* **Returns**
  scaled target.

## maml.utils._signal_processing module

Signal processing utils.

### maml.utils._signal_processing.cwt(z: np.ndarray, widths: np.ndarray, wavelet: str | Callable = ‘morlet2’, \*\*kwargs)

The scalogram of the signal
:param z: 1D signal array
:type z: np.ndarray
:param widths: wavelet widths
:type widths: np.ndarray
:param wavelet: wavelet name
:type wavelet: str

Returns: 2D scalogram.

### maml.utils._signal_processing.fft_magnitude(z: ndarray)

Discrete Fourier Transform the signal z and return
the magnitude of the  coefficients
:param z: 1D signal array
:type z: np.ndarray

Returns: 1D magnitude.

### maml.utils._signal_processing.get_sp_method(sp_method: str | Callable)

Providing a signal processing method name return the callable
:param sp_method: name of the sp function
:type sp_method: str

Returns: callable for signal processing.

### maml.utils._signal_processing.spectrogram(z: np.ndarray, return_time_freq: bool = False)

The spectrogram of the signal
:param z: 1D signal array
:type z: np.ndarray
:param return_time_freq: whether to return time and frequency
:type return_time_freq: bool

Returns: 2D spectrogram.

### maml.utils._signal_processing.wvd(z: np.ndarray, return_all: bool = False)

Wigner Ville Distribution calculator
:param z: signal 1D
:type z: np.ndarray
:param return_all: whether to return time and freq info, default

> only return the wvd information

Returns: NxN wvd matrix.

## maml.utils._stats module

Utils for describers.

### *class* maml.utils._stats.Stats()

Bases: `object`

Calculate the stats of a list of values.
This is particularly helpful when you want to convert
lists of values of different lengths to uniform length
for machine learning purposes.

supported

#### allowed_stats(_ = [‘max’, ‘min’, ‘range’, ‘mode’, ‘mean_absolute_deviation’, ‘mean_absolute_error’, ‘moment’, ‘mean’, ‘inverse_mean’, ‘average’, ‘std’, ‘skewness’, ‘kurtosis’, ‘geometric_mean’, ‘power_mean’, ‘shifted_geometric_mean’, ‘harmonic_mean’_ )

#### *static* average(data: list[float], weights: list[float] | None = None)

Weighted average.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: average value

#### *static* geometric_mean(data: list[float], weights: list[float] | None = None)

Geometric mean of the data.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: geometric mean of the distribution

#### *static* harmonic_mean(data: list[float], weights: list[float] | None = None)

harmonic mean of the data.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: harmonic mean of the distribution

#### *static* inverse_mean(data: list[float], weights: list[float] | None = None)

inverse mean.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: average value

#### *static* kurtosis(data: list[float], weights: list[float] | None = None)

Kurtosis of the distribution.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: Kurtosis of the distribution

#### *static* max(data: list[float], weights: list[float] | None = None)

Max of value
Args:31

> data (list): list of float data
> weights (list): optional weights.

Returns: maximum value

#### *static* mean(data: list[float], weights: list[float] | None = None)

Weighted average.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: average value

#### *static* mean_absolute_deviation(data: list[float], weights: list[float] | None = None)

mean absolute deviation.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*) – optional weights

Returns: mean absolute deviation

#### *static* mean_absolute_error(data: list[float], weights: list[float] | None = None)

mean absolute error.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*) – optional weights

Returns: mean absolute error

#### *static* min(data: list[float], weights: list[float] | None = None)

min of value
:param data: list of float data
:type data: list
:param weights: optional weights.
:type weights: list

Returns: minimum value

#### *static* mode(data: list[float], weights: list[float] | None = None)

Mode of data, if multiple entries have equal counts,
compute the average of those.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*) – optional weights

Returns: mode of values, i.e., max - min

#### *static* moment(data: list[float], weights: list[float] | None = None, order: int | None = None, max_order: int | None = None)

Moment of probability mass function.

order = 1 means weighted mean
order = 2 means standard deviation
order > 2 corresponds to higher order moment to

> the 1./order power
* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data points
  * **order** (*int*) – moment order
  * **max_order** (*int*) – if set, it will overwrite order

Returns: float or list of floats

#### *static* power_mean(data: list[float], weights: list[float] | None = None, p: int = 1)

power mean [https://en.wikipedia.org/wiki/Generalized_mean](https://en.wikipedia.org/wiki/Generalized_mean).

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point
  * **p** (*int*) – power

Returns: power mean of the distribution

#### *static* range(data: list[float], weights: list[float] | None = None)

Range of values
:param data: list of float data
:type data: list
:param weights: optional weights.
:type weights: list

Returns: range of values, i.e., max - min

#### *static* shifted_geometric_mean(data: list[float], weights: list[float] | None = None, shift: float = 100)

Since we cannot calculate the geometric means on negative or zero values,
we can first shift all values to positive and then calculate the geometric mean
afterwards, we shift the computed geometric mean back by a shift value.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point
  * **shift** (*float*) – shift value

Returns: geometric mean of the distribution

#### *static* skewness(data: list[float], weights: list[float] | None = None)

Skewness of the distribution.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: Skewness of the distribution

#### *static* std(data: list[float], weights: list[float] | None = None)

Standard deviation.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point

Returns: Standard deviation

### maml.utils._stats._add_allowed_stats(cls)

Decorate to add allowed_stats to the Stats class.

* **Parameters**
  **cls** – Stats class

Returns: Stats class with allowed_stats attributes

### maml.utils._stats._convert_a_or_b(v: str, a=<class ‘int’>, b=None)

### maml.utils._stats._moment_symbol_conversion(moment_symbol: str)

### maml.utils._stats._root_moment(data, weights, order)

Auxiliary function to compute moment.

* **Parameters**
  * **data** (*list*) – list of float data
  * **weights** (*list*\* or \**None*) – weights for each data point
  * **order** (*int*) – order of moment

Returns: moment of order

### maml.utils._stats.get_full_stats_and_funcs(stats: list)

Get expanded stats function name str and the corresponding
function callables.

* **Parameters**
  **stats** (*list*) – a list of stats names, e.g, [‘mean’, ‘std’, ‘moment:1:None’]

Returns: list of stats names, list of stats callable

### maml.utils._stats.stats_list_conversion(stats_list: list[str])

Convert a list of stats str into a fully expanded list.
This applies mainly to stats that can return a list of values, e.g.,
moment with max_order > 1.

* **Parameters**
  **stats_list** (*list*) – list of stats str

Returns: list of expanded stats str

## maml.utils._tempfile module

Temporary directory and file creation utilities.
This file is adapted from monty.tempfile.

### *class* maml.utils._tempfile.MultiScratchDir(rootpath: str | Path, n_dirs: int = 1, create_symbolic_link: bool = False, copy_from_current_on_enter: bool = False, copy_to_current_on_exit: bool = False)

Bases: `object`

Creates a “with” context manager that automatically handles creation of
temporary directories (utilizing Python’s build in temp directory
functions) and cleanup when done. The main difference between this class
and monty ScratchDir is that multiple temp directories are created here.
It enables the running of multiple jobs simultaneously in the directories
The way it works is as follows:

1. Create multiple temp dirs in specified root path.
2. Optionally copy input files from current directory to temp dir.
3. User loops among all directories
4. User performs specified operations in each directories
5. Change back to original directory.
6. Delete temp dir.

#### SCR_LINK(_ = ‘scratch_link_ )

#### tempdirs(    *: list[str* )

### maml.utils._tempfile._copy_r_with_suffix(src: str, dst: str, suffix: Any | None = None)

Implements a recursive copy function similar to Unix’s “cp -r” command.
Surprisingly, python does not have a real equivalent. shutil.copytree
only works if the destination directory is not present.

* **Parameters**
  * **src** (*str*) – Source folder to copy.
  * **dst** (*str*) – Destination folder.

## maml.utils._typing module

Define several typing for convenient use.

## maml.utils._value_profile module

ValueProfile return values according to certain settings. For
example, one can design a linearly increasing value profile,
a sinusoidal profile or a constant profile, depending on the
step, and previous values.

### *class* maml.utils._value_profile.ConstantValue(value: float, \*\*kwargs)

Bases: `ValueProfile`

Return constant value.

#### get_value()

Return constant value.

### *class* maml.utils._value_profile.LinearProfile(value_start: float, value_end: float = 0.0, max_steps: int = 100, \*\*kwargs)

Bases: `ValueProfile`

LinearProfile by setting starting value and the rate of
value change. The profile can be initialized either by
[value_start, value_end, max_step] or [value_start, rate].

#### get_value()

Get LinearProfile value
Returns: float.

### *class* maml.utils._value_profile.ValueProfile(max_steps: int | None = None, \*\*kwargs)

Bases: `object`

Base class for ValueProfile. The base class has the following methods
.. method:: increment_step(self)

> add one to step

#### get_value(self)

abstract method that return the value.

#### get_value()

abstract method that returns the current value
Returns: value float.

#### increment_step()

Increase step attribute by one.