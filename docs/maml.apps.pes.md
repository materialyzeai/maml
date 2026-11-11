---
layout: default
title: maml.apps.pes.md
nav_exclude: true
---

# maml.apps.pes package

This package contains PotentialMixin classes representing Interatomic Potentials.

## *class* maml.apps.pes.DefectFormation(ff_settings, specie, lattice, alat, \*\*kwargs)

Bases: `LMPStaticCalculator`

Defect formation energy calculator.

### \_parse()

Parse results from dump files.

### \_sanity_check(structure)

Check if the structure is valid for this calculation.

### \_setup()

Setup a calculation, writing input files, etc.

### calculate()

Calculate the vacancy formation given Potential class.

### *static* get_unit_cell(specie, lattice, alat)

Get the unit cell from specie, lattice type and lattice constant.

* **Parameters**
  * **specie** (*str*) – Name of specie.
  * **lattice** (*str*) – The lattice type of structure. e.g. bcc or diamond.
  * **alat** (*float*) – The lattice constant of specific lattice and specie.

## *class* maml.apps.pes.ElasticConstant(ff_settings, potential_type=’external’, deformation_size=1e-06, jiggle=1e-05, maxiter=400, maxeval=1000, full_matrix=False, \*\*kwargs)

Bases: `LMPStaticCalculator`

Elastic constant calculator.

### *RESTART_CONFIG(* = {‘external’: {‘read_command’: ‘read_restart’, ‘restart_file’: ‘restart.equil’, ‘write_command’: ‘write_restart’}, ‘internal’: {‘read_command’: ‘read_restart’, ‘restart_file’: ‘restart.equil’, ‘write_command’: ‘write_restart’}_ )

### \_parse()

Parse results from dump files.

### \_sanity_check(structure)

Check if the structure is valid for this calculation.

### \_setup()

Setup a calculation, writing input files, etc.

## *class* maml.apps.pes.EnergyForceStress(ff_settings, \*\*kwargs)

Bases: `LMPStaticCalculator`

Calculate energy, forces and virial stress of structures.

### \_parse()

Parse results from dump files.

### *static* \_rotate_force_stress(structure, forces, stresses)

### \_sanity_check(structure)

Check if the structure is valid for this calculation.

### \_setup()

Setup a calculation, writing input files, etc.

### calculate(structures)

Calculate the energy, forces and stresses of structures.
Proper rotation of the results are applied when the structure
is triclinic.

* **Parameters**
  **structures** (*list*) – a list of structures

Returns: list of (energy, forces, stresses) tuple

## *class* maml.apps.pes.GAPotential(name=None, param=None)

Bases: `LammpsPotential`

This class implements Smooth Overlap of Atomic Position potentials.

### *abc_impl(* = <_abc.*abc_data object* )

### *static* \_line_up(structure, energy, forces, virial_stress)

Convert input structure, energy, forces, virial_stress to
proper configuration format for MLIP usage.

* **Parameters**
  * **structure** (*Structure*) – Pymatgen Structure object.
  * **energy** (*float*) – DFT-calculated energy of the system.
  * **forces** (*list*) – The forces should have dimension
    (num_atoms, 3).
  * **virial_stress** (*list*) – stress should has 6 distinct
    elements arranged in order [xx, yy, zz, xy, yz, xz].

Returns:

### evaluate(test_structures, test_energies, test_forces, test_stresses=None, predict_energies=True, predict_forces=True, predict_stress=False)

Evaluate energies, forces and stresses of structures with trained
interatomic potentials.

* **Parameters**
  * **test_structures** (    *[**Structure**]*) – List of Pymatgen Structure Objects.
  * **test_energies** (    *[**float**]*) – List of DFT-calculated total energies of
    each structure in structures list.
  * **test_forces** (    *[**np.array**]*) – List of DFT-calculated (m, 3) forces of
    each structure with m atoms in structures list. m can be varied
    with each single structure case.
  * **test_stresses** (*list*) – List of DFT-calculated (6, ) viriral stresses
    of each structure in structures list.
  * **predict_energies** (*bool*) – Whether to predict energies of configurations.
  * **predict_forces** (*bool*) – Whether to predict forces of configurations.
  * **predict_stress** (*bool*) – Whether to predict virial stress of
    configurations.

### *static* from_config(filename)

Initialize potentials with parameters file.

* **Parameters**
  **filename** (*str*) – The file storing parameters of potentials,
  filename should ends with “.xml”.
* **Returns**
  GAPotential.

### pair_coeff(_ = ‘pair_coeff        \* \* {} {} {}_ )

### pair_style(_ = ‘pair_style        quip_ )

### *static* read_cfgs(filename, predict=False)

Read the configuration file.

* **Parameters**
  **filename** (*str*) – The configuration file to be read.

### save(filename=’param.yaml’)

Save parameters of the potentials.

* **Parameters**
  **filename** (*str*) – The file to store parameters of potentials.
* **Returns**
  (str)

### train(train_structures, train_energies, train_forces, train_stresses=None, default_sigma=(0.0005, 0.1, 0.05, 0.01), use_energies=True, use_forces=True, use_stress=False, \*\*kwargs)

Training data with gaussian process regression.

* **Parameters**
  * **train_structures** (    *[**Structure**]*) – The list of Pymatgen Structure object.
    energies ([float]): The list of total energies of each structure
    in structures list.
  * **train_energies** (    *[**float**]*) – List of total energies of each structure in
    structures list.
  * **train_forces** (    *[**np.array**]*) – List of (m, 3) forces array of each structure
    with m atoms in structures list. m can be varied with each
    single structure case.
  * **train_stresses** (*list*) – List of (6, ) virial stresses of each
    structure in structures list.
  * **default_sigma** (*list*) – Error criteria in energies, forces, stress
    and hessian. Should have 4 numbers.
  * **use_energies** (*bool*) – Whether to use dft total energies for training.
    Default to True.
  * **use_forces** (*bool*) – Whether to use dft atomic forces for training.
    Default to True.
  * **use_stress** (*bool*) – Whether to use dft virial stress for training.
    Default to False.
  * **kwargs** – l_max (int): Parameter to configure GAP. The band limit of

spherical harmonics basis function. Default to 12.

n_max (int): Parameter to configure GAP. The number of radial basis

```none
  function. Default to 10.
```

atom_sigma (float): Parameter to configure GAP. The width of gaussian

```none
  atomic density. Default to 0.5.
```

zeta (float): Present when covariance function type is do product.

```none
  Default to 4.
```

cutoff (float): Parameter to configure GAP. The cutoff radius.

```none
  Default to 4.0.
```

cutoff_transition_width (float): Parameter to configure GAP.

```none
  The transition width of cutoff radial. Default to 0.5.
```

delta (float): Parameter to configure Sparsification.

```none
  The signal variance of noise. Default to 1.
```

f0 (float): Parameter to configure Sparsification.

```none
  The signal mean of noise. Default to 0.0.
```

n_sparse (int): Parameter to configure Sparsification.

```none
  Number of sparse points.
```

covariance_type (str): Parameter to configure Sparsification.

```none
  The type of convariance function. Default to dot_product.
```

sparse_method (str): Method to perform clustering in sparsification.

```none
  Default to ‘cur_points’.
```

sparse_jitter (float): Intrisic error of atomic/bond energy,

```none
  used to regularise the sparse covariance matrix.
  Default to 1e-8.
```

e0 (float): Atomic energy value to be subtracted from energies

```none
  before fitting. Default to 0.0.
```

e0_offset (float): Offset of baseline. If zero, the offset is

```none
  the average atomic energy of the input data or the e0
  specified manually. Default to 0.0.
```

### write_cfgs(filename, cfg_pool)

Write the formatted configuration file.

* **Parameters**
  * **filename** (*str*) – The filename to be written.
  * **cfg_pool** (*list*) – The configuration pool contains
    structure and energy/forces properties.

### write_param(xml_filename=’gap.2020.01.xml’)

Write xml file to perform lammps calculation.

* **Parameters**
  **xml_filename** (*str*) – Filename to store xml formatted parameters.

## *class* maml.apps.pes.LMPStaticCalculator(\*\*kwargs)

Bases: `object`

Abstract class to perform static structure property calculation
using LAMMPS.

### *COMMON_CMDS(* = [‘units metal’, ‘atom_style charge’, ‘box tilt large’, ‘read_data data.static’, ‘run 0’_ )

### *abstract* \_parse()

Parse results from dump files.

### *abstract* \_sanity_check(structure)

Check if the structure is valid for this calculation.

### *abstract* \_setup()

Setup a calculation, writing input files, etc.

### allowed_kwargs(_ = [‘lmp_exe’_ )

### calculate(structures)

Perform the calculation on a series of structures.

* **Parameters**
  **structures** – Input structures in a list.
* **Returns**
  List of computed data corresponding to each structure,
  varies with different subclasses.

### set_lmp_exe(lmp_exe: str)

Set lammps executable for the instance.

* **Parameters**
  **lmp_exe** (*str*) – lammps executable path

Returns:

## *class* maml.apps.pes.LatticeConstant(ff_settings, box_relax=True, box_relax_keywords=’aniso 0.0 vmax 0.001’, box_triclinic=False, min_style=’cg’, etol=1e-15, ftol=1e-15, maxiter=5000, maxeval=5000, \*\*kwargs)

Bases: `LMPRelaxationCalculator`

Lattice Constant Relaxation Calculator.

### calculate(structures)

Calculate the relaxed lattice parameters of a list of structures.

* **Parameters**
  **structures** (    *[**Structure**]*) – Input structures in a list.
* **Returns**
  List of relaxed lattice constants (a, b, c in Å) of the input structures.

## *class* maml.apps.pes.MTPotential(name=None, param=None, version=None)

Bases: `LammpsPotential`

This class implements moment tensor potentials.
Installation of the mlip package is needed.
Please refer to [https://mlip.skoltech.ru](https://mlip.skoltech.ru).

### *abc_impl(* = <_abc.*abc_data object* )

### \_line_up(structure, energy, forces, virial_stress)

Convert input structure, energy, forces, virial_stress to
proper configuration format for mlip usage.

* **Parameters**
  * **structure** (*Structure*) – Pymatgen Structure object.
  * **energy** (*float*) – DFT-calculated energy of the system.
  * **forces** (*list*) – The forces should have dimension (num_atoms, 3).
  * **virial_stress** (*list*) – stress should has 6 distinct
    elements arranged in order [xx, yy, zz, yz, xz, xy].

### evaluate(test_structures, test_energies, test_forces, test_stresses=None, \*\*kwargs)

Evaluate energies, forces and stresses of structures with trained
interatomic potentials.

* **Parameters**
  * **test_structures** (    *[**Structure**]*) – List of Pymatgen Structure Objects.
  * **test_energies** (    *[**float**]*) – List of DFT-calculated total energies of
    each structure in structures list.
  * **test_forces** (    *[**np.array**]*) – List of DFT-calculated (m, 3) forces of
    each structure with m atoms in structures list. m can be varied
    with each single structure case.
  * **test_stresses** (*list*) – List of DFT-calculated (6, ) viriral stresses
    of each structure in structures list.
  * **kwargs** – Parameters of write_param method.

### *static* from_config(filename, elements)

Initialize potentials with parameters file.

* **Parameters**
  * **filename** (*str*) – The file storing parameters of potentials, filename should
    ends with “.mtp”.
  * **elements** (*list*) – The list of elements.
* **Returns**
  MTPotential

### pair_coeff(_ = ‘pair_coeff        \* \*_ )

### pair_style(_ = ‘pair_style        mlip {}_ )

### read_cfgs(filename)

* **Parameters**
  **filename** (*str*) – The configuration file to be read.

### train(train_structures, train_energies, train_forces, train_stresses, unfitted_mtp=’08g.mtp’, max_dist=5, radial_basis_size=8, max_iter=1000, energy_weight=1, force_weight=0.01, stress_weight=0.001, init_params=’same’, scale_by_force=0, bfgs_conv_tol=0.001, weighting=’vibration’)

Training data with moment tensor method.

* **Parameters**
  * **train_structures** (    *[**Structure**]*) – The list of Pymatgen Structure object.
    energies ([float]): The list of total energies of each structure
    in structures list.
  * **train_energies** (    *[**float**]*) – List of total energies of each structure in
    structures list.
  * **train_forces** (    *[**np.array**]*) – List of (m, 3) forces array of each structure
    with m atoms in structures list. m can be varied with each single
    structure case.
  * **train_stresses** (*list*) – List of (6, ) virial stresses of each structure
    in structures list.
  * **unfitted_mtp** (*str*) – Define the initial mtp file. Default to the mtp file
    stored in .params directory.
  * **max_dist** (*float*) – The actual radial cutoff.
  * **radial_basis_size** (*int*) – Relevant to number of radial basis function.
  * **max_iter** (*int*) – The number of maximum iteration.
  * **energy_weight** (*float*) – The weight of energy.
  * **force_weight** (*float*) – The weight of forces.
  * **stress_weight** (*float*) – The weight of stresses. Zero-weight can be assigned.
  * **init_params** (*str*) – How to initialize parameters if a potential was not
    pre-fitted. Choose from “same” and “random”.
  * **scale_by_force** (*float*) – Default=0. If >0 then configurations near equilibrium
    (with roughly force < scale_by_force) get more weight.
  * **bfgs_conv_tol** (*float*) – Stop training if error dropped by a factor smaller than this
    over 50 BFGS iterations.
  * **weighting** (*str*) – How to weight configuration with different sizes relative to each other.
    Choose from “vibrations”, “molecules” and “structures”.

### write_cfg(filename, cfg_pool)

Write configurations to file
:param filename: filename
:type filename: str
:param cfg_pool: list of configurations.
:type cfg_pool: list

Returns:

### write_ini(mtp_filename=’fitted.mtp’, select=False, \*\*kwargs)

Write mlip.ini file for mlip packages of version mlip-2 or mlip-dev.
Supported keyword arguments are parallel with options stated in the mlip manuals.
mlip-2 is recommended, as it is the only officially supported version by mlip.
Please refer to [https://mlip.skoltech.ru](https://mlip.skoltech.ru).

* **Parameters**
  * **mlip-2** – mtp_filename (str): Name of file with MTP to be loaded.
    write_cfgs (str): Name of file for mlp processed configurations to be written to.
    write_cfgs_skip (int): Skipped number of processed configurations before writing.
    select (bool): activates or deactivates calculation of extrapolation grades and

> optionally writing configurations with high extrapolation grades. False is
> recommended for large-scale MD run.

select_save_selected (str): Name of file for saving configurations with grade

```none
  exceeding select_threshold.
```

select_threshold (float): Configurations with extrapolation grade exceeding the

```none
  value will be saved to the specified file.
```

select_threshold_break (float): The mlip execution will be interrupted if the

```none
  extrapolation grade exceeds this value.
```

select_load_state (str): Name of file for loading the active learning state,

```none
  typically created by the mlp calc-grade command.
```

select_log (str): Name of file (or standard output stream stdout/stderr) for

```none
  writing a log of the configuration selection process.
```

```none
* **mlip-dev** – Abinitio (int): Defines Ab-initio models. Default to 1.
```

> 0: If Ab-initio models is not required.
> 1: Used if driver provides EFS data with configurations.
> 2: Use embedded Lennard-Jones pair potentials.
> > r_min (float): Distance to minimum of pair function (in Angstroms).
> > ```none
> > Default to 2.0.
> > ```
> > scale (float): Value of pair function in minimum (in eV).
> > ```none
> > Default to 1.0.
> > ```
> > cutoff (float): Cutoff radius (in Angstroms). Default to 5.0.
> > 3: Use DFT models by VASP. Linking via files exchange.
> > POSCAR (str): Relative path of POSCAR file.
> > OUTCAR (str): Relative path of OUTCAR file.
> > Start_command (str): Relative path of command file.
> > 4: Use potentials calculating by LAMMPS. Linking via files exchange.
> > Input_file (str): File with configuration to be read by lammps.
> > Output_file (str): File with configuration and EFS data to be read by MLIP.
> > Start_command (str): Relative path of command file.
> > 5: Use MTP as Ab-initio potentials.
> > MTP_filename (str): MTP file name.

MLIP (str): MTP.

> load_from (str): Potential filename.
> Cacluate_EFS (bool): Whether to perform EFS calculation by MTP.
> Fit (bool): Whether to perform MTP learning.
> > Save (str): Output MTP file name (for trained MTP).
> > Energy_equation_weight (float): Weight for energy equation in
> > > fitting procedure. Default to 1.0.
> > > Forces_equation_weight (float): Weight for forces equations in
> > ```none
> > fitting procedure. Default to 0.001.
> > ```
> > Stress_equation_weight (float): Weight for stresses equations in
> > ```none
> > fitting procedure.  Default to 0.1.
> > ```
> > Relative_forces_weight (float): If greater than zero, large forces
> > ```none
> > will be fitted less accurate than small. Default to 0.0.
> > ```
> > Fit_log (str): File to write fitting log. No logging if not specified.
> > ```none
> > Default to None.
> > ```
> Select (bool): Whether to activate active learning. Default to False.
> > Site_E_weight (float): Weight for site energy equations in
> > ```none
> > selection procedure. Default to 1.0.
> > ```
> > Energy_weight (float): Weight for energy equation in
> > ```none
> > selection procedure. Default to 0.0.
> > ```
> > Forces_weight (float): Weight for forces equations in
> > ```none
> > selection procedure. Default to 0.0.
> > ```
> > Stress_weight (float): Weight for stresses equations in
> > ```none
> > selection procedure. Default to 0.0.
> > ```
> > Threshold_slct (float): Selection threshold - maximum
> > ```none
> > allowed extrapolation level. Default to 0.1.
> > ```
> > Save_TS (str): Filename where selected configurations
> > ```none
> > will be saved. No configuration saving if not specified.
> > Default to None.
> > ```
> > Save_state (str): Filename where state of the selection
> > ```none
> > will be saved. No saving if not specified. Default to None.
> > ```
> > Load_state (str): Filename where state of the selection
> > ```none
> > will be loaded. No saving if not specified. Default to None.
> > ```
> > Select_log (str): File to write fitting log. No logging
> > ```none
> > if not specified. Default to None.
> > ```
> LOFT (bool): Whether to perform learning on the fly. Default to False
> > EFSviaMTP (bool): Works only on LOFT regime. If True,
> > ```none
> > only MTP-calculated EFS will be passed to driver, else
> > pass to driver ab-initio EFS while LOTF when learning occurs.
> > ```
> > Log (str): Filename to write log of learning on the fly process.
> > ```none
> > No logging if not specified. Default to None.
> > ```
> Check_errors (bool): If True, comparison and accumulation of
> ```none
> error statistics for EFS calculated by ab-initio models and MTP.
> Default to False.
> ```
> ```none
> Log (str): Filename to write log of learning on the fly process.
> ```
> ```none
>     No logging if not specified. Default to None.
> ```
> Write_cfgs (bool): File for writing all processed configurations.
> ```none
> No configuration recording if not specified. Default to None.
> ```
> ```none
> Skip_N (int): The number of configurations to skip while writing.
> ```
> ```none
>     Default to 0.
> ```
> Log (str): Filename to write MLIP log. No logging if not specified.
> ```none
> Default to None.
> ```

Driver (int): Defines the configuration driver. Default to 1.

> 0: No driver or external MD driver.
> 1: Read configurations from database file.
> > Database_filename (str): Configuration file name.
> > Max_count (int): Maximal number of configurations to read.
> > Log (str): Filename to write reading log. No logging
> > > if not specified. Default to None.
> > > 2: Embedded algorithm for relaxation.
> > > Pressure (float): External pressure (in GPa).
> > ```none
> > If not zero enthalpy is minimized. Default to 0.0.
> > ```
> > Iteration_limit (int): Maximal number of iteration of
> > ```none
> > the relaxation process. Default to 500.
> > ```
> > Min_dist (float): Minimal interatomic distance constraint
> > ```none
> > (in Angstroms). Default to 1.0.
> > ```
> > Forces_tolerance (float): Forces on atoms in relaxed
> > ```none
> > configuration should be smaller than this value
> > (in eV/Angstroms). Default to 0.0001.
> > ```
> > Stress_tolerance (float): Stresses in relaxed configuration
> > ```none
> > should be smaller than this value (in GPa). Default to 0.001.
> > ```
> > Max_step (float): Maximal allowed displacement of atoms and
> > ```none
> > lattice vectors in Cartesian coordinates (in Angstroms).
> > Default to 0.5.
> > ```
> > Min_step (float): Minimal displacement of atoms and
> > ```none
> > lattice vectors in Cartesian coordinates (in Angstroms).
> > Default to 1.0e-8.
> > ```
> > BFGS_Wolfe_C1 (float): Wolfe condition constant on the function
> > ```none
> > decrease (linesearch stopping criterea). Default to 1.0e-3.
> > ```
> > BFGS_Wolfe_C2 (float): Wolfe condition constant on the gradient
> > ```none
> > decrease (linesearch stopping criterea). Default to 0.7.
> > ```
> > Save_relaxed (str): Filename for output results of relaxation.
> > ```none
> > No configuration will be saved if not specified.
> > Default to None.
> > ```
> > Log (str): Filename to write relaxation log. No logging
> > ```none
> > if not specified. Default to None.
> > ```

### write_param(fitted_mtp=’fitted.mtp’, \*\*kwargs)

Write fitted mtp parameter file to perform lammps calculation.

* **Parameters**
  **fitted_mtp** (*str*) – Filename to store xml formatted parameters.

## *class* maml.apps.pes.NNPotential(name=None, param=None, weight_param=None, scaling_param=None)

Bases: `LammpsPotential`

This class implements Neural Network Potential.

### *abc_impl(* = <_abc.*abc_data object* )

### \_line_up(structure, energy, forces, virial_stress)

Convert input structure, energy, forces, virial_stress to
proper configuration format for n2p2 usage. Note that
n2p2 takes bohr as length unit and Hartree as energy unit.

* **Parameters**
  * **structure** (*Structure*) – Pymatgen Structure object.
  * **energy** (*float*) – DFT-calculated energy of the system.
  * **forces** (*list*) – The forces should have dimension
    (num_atoms, 3).
  * **virial_stress** (*list*) – stress should has 6 distinct
    elements arranged in order [xx, yy, zz, xy, yz, xz].

### bohr_to_angstrom(_ = 0.52917721090_ )

### eV_to_Ha(_ = 0.03674932217565_ )

### evaluate(test_structures, test_energies, test_forces, test_stresses=None)

Evaluate energies, forces and stresses of structures with trained
interatomic potentials.

* **Parameters**
  * **test_structures** (    *[**Structure**]*) – List of Pymatgen Structure Objects.
  * **test_energies** (    *[**float**]*) – List of DFT-calculated total energies of
    each structure in structures list.
  * **test_forces** (    *[**np.array**]*) – List of DFT-calculated (m, 3) forces of
    each structure with m atoms in structures list. m can be varied
    with each single structure case.
  * **test_stresses** (*list*) – List of DFT-calculated (6, ) viriral stresses
    of each structure in structures list.

### *static* from_config(input_filename, scaling_filename, weights_filenames)

Initialize potentials with parameters file.

* **Parameters**
  * **input_filename** (*str*) – The file storing the input configuration of
    Neural Network Potential.
  * **scaling_filename** (*str*) – The file storing scaling info of
    Neural Network Potential.
  * **weights_filenames** (*list*) – List of files storing weights of each specie in
    Neural Network Potential.

### load_input(filename=’input.nn’)

Load input file from trained Neural Network Potential.

* **Parameters**
  **filename** (*str*) – The input filename.

### load_scaler(scaling_filename)

Load scaling info of trained Neural Network Potential.

* **Parameters**
  **scaling_filename** (*str*) – The scaling file.

### load_weights(weights_filename, specie)

Load weights file of trained Neural Network Potential.

* **Parameters**
  * **weights_filename** (*str*) – The weights file.
  * **specie** (*str*) – The name of specie.

### pair_coeff(_ = ‘pair_coeff        \* \* {}_ )

### pair_style(_ = ‘pair_style        nnp dir “./” showew no showewsum 0 maxew 10000000 resetew yes cflength 1.8897261328 cfenergy 0.0367493254_ )

### read_cfgs(filename=’output.data’)

Read the configuration file.

* **Parameters**
  **filename** (*str*) – The configuration file to be read.

### train(train_structures, train_energies, train_forces, train_stresses=None, \*\*kwargs)

Training data with moment tensor method.

* **Parameters**
  * **train_structures** (    *[**Structure**]*) – The list of Pymatgen Structure object.
    energies ([float]): The list of total energies of each structure
    in structures list.
  * **train_energies** (    *[**float**]*) – List of total energies of each structure in
    structures list.
  * **train_forces** (    *[**np.array**]*) – List of (m, 3) forces array of each structure
    with m atoms in structures list. m can be varied with each
    single structure case.
  * **train_stresses** (*list*) – List of (6, ) virial stresses of each
    structure in structures list.
  * **kwargs** – Parameters in write_input method.

### write_cfgs(filename, cfg_pool)

Write the formatted configuration file.

* **Parameters**
  * **filename** (*str*) – The filename to be written.
  * **cfg_pool** (*list*) – The configuration pool contains
    structure and energy/forces properties.

### write_input(\*\*kwargs)

Write input.nn file to train the Neural Network Potential.

* **Parameters**
  * **atom_energy** (*float*) – Atomic reference energy.
  * **kwargs** – General nnp settings:

atom_energy (dict): Free atom reference energy for each specie.
cutoff_type (int): Type of cutoff function. Default to 1

> (i.e., cosine function).

scale_features (int): Determine the method to scale the

```none
  symmetry function.
  0: no scaling.
  1: scale_symmetry_functions.
  2: center_symmetry_functions.
  3. scale_symmetry_functions_sigma.
```

scale_min_short (float): Minimum value for scaling.

```none
  Default to 0.0.
```

scale_max_short (float): Maximum value for scaling.

```none
  Default to 1.
```

hidden_layers (list): List of the numbers of

```none
  nodes in each hidden layer.
```

activations (str): Activation function for each hidden layer.

```none
  ’t’: tanh, ‘s’: logistic, ‘p’: softplus.
```

normalize_nodes (boolean): Whether to normalize input of nodes.

Additional settings for training:

```none
  epoch (int): Number of training epochs.
  updater_type (int): Weight update method

  > 0: gradient Descent, 1: Kalman filter.

  parallel_mode (int): Training parallelization used.

      Default to serial mode.

  jacobian_mode (int): Jacobian computation mode.

      0: Summation to single gradient,
      1: Per-task summed gradient,
      2: Full Jacobian.

  update_strategy (int): Update strategy.

      0: combined, 1: per-element.

  selection_mode (int): Update candidate selection mode.

      0: random, 1: sort, 2: threshold

  task_batch_size_energy (int): Number of energy update

      candidates prepared per task for each update.

  task_batch_size_force (int): Number of force update

      candidates prepared per task for each update.

  test_fraction (float): Fraction of structures kept for

      testing.

  force_weight (float): Weight of force updates relative

      to energy updates. Default to 10.0

  short_energy_fraction (float): Fraction of energy updates

      per epoch. Default to 1.0.

  short_force_fraction (float): Fraction of force updates

      per epoch. Default to 0.02315.

  short_energy_error_threshold (float): RMSE threshold for

      energy update candidates. Default to 0.0.

  short_force_error_threshold (float): RMSE threshold for

      force update candidates. Default to 1.0.

  rmse_threshold_trials (int): Maximum number of RMSE

      threshold trials. Default to 3.

  weights_min (float): Minimum value for initial random

      weights. Default to -1.

  weights_max (float): Maximum value for initial random

      weights. Default to 1.

  write_trainpoints (int): Write energy comparison every

      this many epochs. Default to 1.

  write_trainforces (int): Write force comparison every

      this many epochs. Default to 1.

  write_weights_epoch (int): Write weights every this many

      epochs. Default to 1.

  write_neuronstats (int): Write neuron statistics every

      this many epochs. Default to 1.

  # Kalman Filter
  kalman_type (int): Kalman filter type. Default to 0.
  kalman_epsilon (float): General Kalman filter parameter

  > epsilon. Default to 0.01.

  kalman_q0 (float): General Kalman filter parameter q0.

      Default to 0.01.

  kalman_qtau (float): General Kalman filter parameter

      qtau. Default to 2.302.

  kalman_qmin (float): General Kalman filter parameter qmin.

      Default to 1e-6.

  kalman_eta (float): Standard Kalman filter parameter eta.

      Default to 0.01.

  kalman_etatau (float): Standard Kalman filter parameter

      etatau. Defaults to 2.302.

  kalman_etamax (float): Standard Kalman filter parameter

      etamax. Default to 1.0.
```

Symmetry functions:

```none
  r_cut (float): Cutoff distance (unit: Å).
  r_etas (numpy.array): η in radial function.
  r_shift (numpy.array): Rs in radial function.
  a_etas (numpy.array): η in angular function.
  zetas (numpy.array): ζ in angular function.
  lambdas (numpy.array): λ in angular function. Default to (1, -1).
```

### write_param()

Write optimized weights file to perform energy and force prediction.

## *class* maml.apps.pes.NudgedElasticBand(ff_settings, specie, lattice, alat, num_replicas=7, \*\*kwargs)

Bases: `LMPStaticCalculator`

NudgedElasticBand migration energy calculator.

### \_parse()

Parse results from dump files.

### \_sanity_check(structure)

Check if the structure is valid for this calculation.

### \_setup()

Setup a calculation, writing input files, etc.

### calculate()

Calculate the NEB barrier given Potential class.

### *static* get_unit_cell(specie, lattice, alat)

Get the unit cell from specie, lattice type and lattice constant.

* **Parameters**
  * **specie** (*str*) – Name of specie.
  * **lattice** (*str*) – The lattice type of structure. e.g. bcc or diamond.
  * **alat** (*float*) – The lattice constant of specific lattice and specie.

## *class* maml.apps.pes.Potential(model, describer: BaseDescriber | None = None, \*\*kwargs)

Bases: `PotentialMixin`, `BaseModel`

Potential models that can be used to fit structure-[energy, force, stress]
pairs.

## *class* maml.apps.pes.SNAPotential(model, name=None)

Bases: `LammpsPotential`

This class implements Spectral Neighbor Analysis Potential.

### *abc_impl(* = <_abc.*abc_data object* )

### evaluate(test_structures, test_energies, test_forces, test_stresses=None, include_stress=False, stress_format=’VASP’)

Evaluate energies, forces and stresses of structures with trained
machinea learning potentials.

* **Parameters**
  * **test_structures** (    *[**Structure**]*) – List of Pymatgen Structure Objects.
  * **test_energies** (    *[**float**]*) – List of DFT-calculated total energies of
    each structure in structures list.
  * **test_forces** (    *[**np.array**]*) – List of DFT-calculated (m, 3) forces of
    each structure with m atoms in structures list. m can be varied
    with each single structure case.
  * **test_stresses** (*list*) – List of DFT-calculated (6, ) viriral stresses
    of each structure in structures list.
  * **include_stress** (*bool*) – Whether to include stress components.
  * **stress_format** (*str*) – stress format, default to “VASP”

### *static* from_config(param_file, coeff_file, \*\*kwargs)

Initialize potentials with parameters file and coefficient file.

* **Parameters**
  * **param_file** (*str*) – The file storing the configuration of potentials.
  * **coeff_file** (*str*) – The file storing the coefficients of potentials.
* **Returns**
  SNAPotential.

### pair_coeff(_ = ‘pair_coeff        \* \* {coeff_file} {param_file} {elements}_ )

### pair_style(_ = ‘pair_style        snap_ )

### train(train_structures, train_energies, train_forces, train_stresses=None, include_stress=False, stress_format=’VASP’, \*\*kwargs)

Training data with models.

* **Parameters**
  * **train_structures** (    *[**Structure**]*) – The list of Pymatgen Structure object.
    energies ([float]): The list of total energies of each structure
    in structures list.
  * **train_energies** (    *[**float**]*) – List of total energies of each structure in
    structures list.
  * **train_forces** (    *[**np.array**]*) – List of (m, 3) forces array of each
    structure with m atoms in structures list. m can be varied with
    each single structure case.
  * **train_stresses** (*list*) – List of (6, ) virial stresses of each
    structure in structures list.
  * **include_stress** (*bool*) – Whether to include stress components.
  * **stress_format** (*string*) – stress format, default to VASP

### write_param()

Write parameter and coefficient file to perform lammps calculation.

## *class* maml.apps.pes.SpectralNeighborAnalysis(rcutfac, twojmax, element_profile, quadratic=False, \*\*kwargs)

Bases: `LMPStaticCalculator`

Calculator for bispectrum components to characterize the local
neighborhood of each atom in a general way.

Usage:

```none
[(b, db, e)] = sna.calculate([Structure])
b: 2d NumPy array with shape (N, n_bs) containing bispectrum

> coefficients, where N is the No. of atoms in structure and
> n_bs is the No. of bispectrum components.

db: 2d NumPy array with shape (N, 3 \* n_bs \* n_elements)

    containing the first order derivatives of bispectrum
    coefficients with respect to atomic coordinates,
    where n_elements is the No. of elements in element_profile.

e: 2d NumPy array with shape (N, 1) containing the element of

    each atom.
```

### *CMDS(* = [‘pair_style lj/cut 10’, ‘pair_coeff \* \* 1 1’, ‘compute sna all sna/atom ‘, ‘compute snad all snad/atom ‘, ‘compute snav all snav/atom ‘, ‘dump 1 all custom 1 dump.element element’, ‘dump_modify 1 sort id’, ‘dump 2 all custom 1 dump.sna c_sna[\*]’, ‘dump_modify 2 sort id’, ‘dump 3 all custom 1 dump.snad c_snad[\*]’, ‘dump_modify 3 sort id’, ‘dump 4 all custom 1 dump.snav c_snav[\*]’, ‘dump_modify 4 sort id’_ )

### \_parse()

Parse results from dump files.

### \_sanity_check(structure)

Check if the structure is valid for this calculation.

### \_setup()

Setup a calculation, writing input files, etc.

### *static* get_bs_subscripts(twojmax)

Method to list the subscripts 2j1, 2j2, 2j of bispectrum
components.

* **Parameters**
  **twojmax** (*int*) – Band limit for bispectrum components.
* **Returns**
  List of all subscripts [2j1, 2j2, 2j].

### *property* n_bs()

Returns No. of bispectrum components to be calculated.

## maml.apps.pes.get_default_lmp_exe()

Get lammps executable
Returns: Lammps executable name.

## maml.apps.pes._base module

Base classes for potentials.

### *class* maml.apps.pes._base.Potential(model, describer: BaseDescriber | None = None, \*\*kwargs)

Bases: `PotentialMixin`, `BaseModel`

Potential models that can be used to fit structure-[energy, force, stress]
pairs.

### *class* maml.apps.pes._base.PotentialMixin()

Bases: `object`

Abstract base class for a Interatomic Potential.

#### *abstract* evaluate(test_structures, test_energies, test_forces, test_stresses)

Evaluate energies, forces and stresses of structures with trained
interatomic potentials.

* **Parameters**
  * **test_structures** (*list*) – List of Pymatgen Structure Objects.
  * **test_energies** (*list*) – List of DFT-calculated total energies of each
    structure in structures list.
  * **test_forces** (*list*) – List of DFT-calculated (m, 3) forces of each
    structure with m atoms in structures list. m can be varied with
    each single structure case.
  * **test_stresses** (*list*) – List of DFT-calculated (6, ) viriral stresses of
    each structure in structures list.

#### *abstract* from_config(\*\*kwargs)

Initialize potentials with parameters file.

#### *abstract* predict_efs(structure)

Predict energy, forces and stresses of the structure.

* **Parameters**
  **structure** (*Structure*) – Pymatgen Structure object.
* **Returns**
  energy, forces, stress

#### *abstract* train(train_structures, train_energies, train_forces, train_stresses, \*\*kwargs)

Train interatomic potentials with energies, forces ann stresses corresponding

```none
to structures.
```

* **Parameters**
  * **train_structures** (*list*) – List of Pymatgen Structure objects.
  * **train_energies** (*list*) – List of DFT-calculated total energies of each
    structure in structures list.
  * **train_forces** (*list*) – List of DFT-calculated (m, 3) forces of each
    structure with m atoms in structures list. m can be varied with
    each single structure case.
  * **train_stresses** (*list*) – List of DFT-calculated (6, ) virial stresses of
    each structure in structures list.

#### *abstract* write_param()

Write optimized weights file to perform energy and force prediction via
lammps calculation.

## maml.apps.pes._gap module

This module provides SOAP-GAP interatomic potential class.

### *class* maml.apps.pes._gap.GAPotential(name=None, param=None)

Bases: `LammpsPotential`

This class implements Smooth Overlap of Atomic Position potentials.

#### *abc_impl(* = <_abc.*abc_data object* )

#### *static* \_line_up(structure, energy, forces, virial_stress)

Convert input structure, energy, forces, virial_stress to
proper configuration format for MLIP usage.

* **Parameters**
  * **structure** (*Structure*) – Pymatgen Structure object.
  * **energy** (*float*) – DFT-calculated energy of the system.
  * **forces** (*list*) – The forces should have dimension
    (num_atoms, 3).
  * **virial_stress** (*list*) – stress should has 6 distinct
    elements arranged in order [xx, yy, zz, xy, yz, xz].

Returns:

#### evaluate(test_structures, test_energies, test_forces, test_stresses=None, predict_energies=True, predict_forces=True, predict_stress=False)

Evaluate energies, forces and stresses of structures with trained
interatomic potentials.

* **Parameters**
  * **test_structures** (    *[**Structure**]*) – List of Pymatgen Structure Objects.
  * **test_energies** (    *[**float**]*) – List of DFT-calculated total energies of
    each structure in structures list.
  * **test_forces** (    *[**np.array**]*) – List of DFT-calculated (m, 3) forces of
    each structure with m atoms in structures list. m can be varied
    with each single structure case.
  * **test_stresses** (*list*) – List of DFT-calculated (6, ) viriral stresses
    of each structure in structures list.
  * **predict_energies** (*bool*) – Whether to predict energies of configurations.
  * **predict_forces** (*bool*) – Whether to predict forces of configurations.
  * **predict_stress** (*bool*) – Whether to predict virial stress of
    configurations.

#### *static* from_config(filename)

Initialize potentials with parameters file.

* **Parameters**
  **filename** (*str*) – The file storing parameters of potentials,
  filename should ends with “.xml”.
* **Returns**
  GAPotential.

#### pair_coeff(_ = ‘pair_coeff        \* \* {} {} {}_ )

#### pair_style(_ = ‘pair_style        quip_ )

#### *static* read_cfgs(filename, predict=False)

Read the configuration file.

* **Parameters**
  **filename** (*str*) – The configuration file to be read.

#### save(filename=’param.yaml’)

Save parameters of the potentials.

* **Parameters**
  **filename** (*str*) – The file to store parameters of potentials.
* **Returns**
  (str)

#### train(train_structures, train_energies, train_forces, train_stresses=None, default_sigma=(0.0005, 0.1, 0.05, 0.01), use_energies=True, use_forces=True, use_stress=False, \*\*kwargs)

Training data with gaussian process regression.

* **Parameters**
  * **train_structures** (    *[**Structure**]*) – The list of Pymatgen Structure object.
    energies ([float]): The list of total energies of each structure
    in structures list.
  * **train_energies** (    *[**float**]*) – List of total energies of each structure in
    structures list.
  * **train_forces** (    *[**np.array**]*) – List of (m, 3) forces array of each structure
    with m atoms in structures list. m can be varied with each
    single structure case.
  * **train_stresses** (*list*) – List of (6, ) virial stresses of each
    structure in structures list.
  * **default_sigma** (*list*) – Error criteria in energies, forces, stress
    and hessian. Should have 4 numbers.
  * **use_energies** (*bool*) – Whether to use dft total energies for training.
    Default to True.
  * **use_forces** (*bool*) – Whether to use dft atomic forces for training.
    Default to True.
  * **use_stress** (*bool*) – Whether to use dft virial stress for training.
    Default to False.
  * **kwargs** – l_max (int): Parameter to configure GAP. The band limit of

spherical harmonics basis function. Default to 12.

n_max (int): Parameter to configure GAP. The number of radial basis

```none
  function. Default to 10.
```

atom_sigma (float): Parameter to configure GAP. The width of gaussian

```none
  atomic density. Default to 0.5.
```

zeta (float): Present when covariance function type is do product.

```none
  Default to 4.
```

cutoff (float): Parameter to configure GAP. The cutoff radius.

```none
  Default to 4.0.
```

cutoff_transition_width (float): Parameter to configure GAP.

```none
  The transition width of cutoff radial. Default to 0.5.
```

delta (float): Parameter to configure Sparsification.

```none
  The signal variance of noise. Default to 1.
```

f0 (float): Parameter to configure Sparsification.

```none
  The signal mean of noise. Default to 0.0.
```

n_sparse (int): Parameter to configure Sparsification.

```none
  Number of sparse points.
```

covariance_type (str): Parameter to configure Sparsification.

```none
  The type of convariance function. Default to dot_product.
```

sparse_method (str): Method to perform clustering in sparsification.

```none
  Default to ‘cur_points’.
```

sparse_jitter (float): Intrisic error of atomic/bond energy,

```none
  used to regularise the sparse covariance matrix.
  Default to 1e-8.
```

e0 (float): Atomic energy value to be subtracted from energies

```none
  before fitting. Default to 0.0.
```

e0_offset (float): Offset of baseline. If zero, the offset is

```none
  the average atomic energy of the input data or the e0
  specified manually. Default to 0.0.
```

#### write_cfgs(filename, cfg_pool)

Write the formatted configuration file.

* **Parameters**
  * **filename** (*str*) – The filename to be written.
  * **cfg_pool** (*list*) – The configuration pool contains
    structure and energy/forces properties.

#### write_param(xml_filename=’gap.2020.01.xml’)

Write xml file to perform lammps calculation.

* **Parameters**
  **xml_filename** (*str*) – Filename to store xml formatted parameters.

## maml.apps.pes._lammps module

This module provides basic LAMMPS calculator classes.

### *class* maml.apps.pes._lammps.DefectFormation(ff_settings, specie, lattice, alat, \*\*kwargs)

Bases: `LMPStaticCalculator`

Defect formation energy calculator.

#### \_parse()

Parse results from dump files.

#### \_sanity_check(structure)

Check if the structure is valid for this calculation.

#### \_setup()

Setup a calculation, writing input files, etc.

#### calculate()

Calculate the vacancy formation given Potential class.

#### *static* get_unit_cell(specie, lattice, alat)

Get the unit cell from specie, lattice type and lattice constant.

* **Parameters**
  * **specie** (*str*) – Name of specie.
  * **lattice** (*str*) – The lattice type of structure. e.g. bcc or diamond.
  * **alat** (*float*) – The lattice constant of specific lattice and specie.

### *class* maml.apps.pes._lammps.ElasticConstant(ff_settings, potential_type=’external’, deformation_size=1e-06, jiggle=1e-05, maxiter=400, maxeval=1000, full_matrix=False, \*\*kwargs)

Bases: `LMPStaticCalculator`

Elastic constant calculator.

#### *RESTART_CONFIG(* = {‘external’: {‘read_command’: ‘read_restart’, ‘restart_file’: ‘restart.equil’, ‘write_command’: ‘write_restart’}, ‘internal’: {‘read_command’: ‘read_restart’, ‘restart_file’: ‘restart.equil’, ‘write_command’: ‘write_restart’}_ )

#### \_parse()

Parse results from dump files.

#### \_sanity_check(structure)

Check if the structure is valid for this calculation.

#### \_setup()

Setup a calculation, writing input files, etc.

### *class* maml.apps.pes._lammps.EnergyForceStress(ff_settings, \*\*kwargs)

Bases: `LMPStaticCalculator`

Calculate energy, forces and virial stress of structures.

#### \_parse()

Parse results from dump files.

#### *static* \_rotate_force_stress(structure, forces, stresses)

#### \_sanity_check(structure)

Check if the structure is valid for this calculation.

#### \_setup()

Setup a calculation, writing input files, etc.

#### calculate(structures)

Calculate the energy, forces and stresses of structures.
Proper rotation of the results are applied when the structure
is triclinic.

* **Parameters**
  **structures** (*list*) – a list of structures

Returns: list of (energy, forces, stresses) tuple

### *class* maml.apps.pes._lammps.LMPRelaxationCalculator(ff_settings, box_relax=True, box_relax_keywords=’aniso 0.0 vmax 0.001’, box_triclinic=False, min_style=’cg’, etol=1e-15, ftol=1e-15, maxiter=5000, maxeval=5000, \*\*kwargs)

Bases: `LMPStaticCalculator`

Structural Relaxation Calculator.

#### \_parse()

Parse results from dump files.

#### \_sanity_check(structure)

Check if the structure is valid for this calculation.

#### \_setup()

Setup a calculation, writing input files, etc.

### *class* maml.apps.pes._lammps.LMPStaticCalculator(\*\*kwargs)

Bases: `object`

Abstract class to perform static structure property calculation
using LAMMPS.

#### *COMMON_CMDS(* = [‘units metal’, ‘atom_style charge’, ‘box tilt large’, ‘read_data data.static’, ‘run 0’_ )

#### *abstract* \_parse()

Parse results from dump files.

#### *abstract* \_sanity_check(structure)

Check if the structure is valid for this calculation.

#### *abstract* \_setup()

Setup a calculation, writing input files, etc.

#### allowed_kwargs(_ = [‘lmp_exe’_ )

#### calculate(structures)

Perform the calculation on a series of structures.

* **Parameters**
  **structures** – Input structures in a list.
* **Returns**
  List of computed data corresponding to each structure,
  varies with different subclasses.

#### set_lmp_exe(lmp_exe: str)

Set lammps executable for the instance.

* **Parameters**
  **lmp_exe** (*str*) – lammps executable path

Returns:

### *class* maml.apps.pes._lammps.LammpsPotential(model, describer: BaseDescriber | None = None, \*\*kwargs)

Bases: `Potential`, `ABC`

Lammps compatible potentials that call lammps executable for
energy/force/stress calculations.

#### *abc_impl(* = <_abc.*abc_data object* )

#### predict_efs(structure)

Predict energy, forces and stresses of the structure.

* **Parameters**
  **structure** (*Structure*) – Pymatgen Structure object.
* **Returns**
  energy, forces, stress

### *class* maml.apps.pes._lammps.LatticeConstant(ff_settings, box_relax=True, box_relax_keywords=’aniso 0.0 vmax 0.001’, box_triclinic=False, min_style=’cg’, etol=1e-15, ftol=1e-15, maxiter=5000, maxeval=5000, \*\*kwargs)

Bases: `LMPRelaxationCalculator`

Lattice Constant Relaxation Calculator.

#### calculate(structures)

Calculate the relaxed lattice parameters of a list of structures.

* **Parameters**
  **structures** (    *[**Structure**]*) – Input structures in a list.
* **Returns**
  List of relaxed lattice constants (a, b, c in Å) of the input structures.

### *class* maml.apps.pes._lammps.NudgedElasticBand(ff_settings, specie, lattice, alat, num_replicas=7, \*\*kwargs)

Bases: `LMPStaticCalculator`

NudgedElasticBand migration energy calculator.

#### \_parse()

Parse results from dump files.

#### \_sanity_check(structure)

Check if the structure is valid for this calculation.

#### \_setup()

Setup a calculation, writing input files, etc.

#### calculate()

Calculate the NEB barrier given Potential class.

#### *static* get_unit_cell(specie, lattice, alat)

Get the unit cell from specie, lattice type and lattice constant.

* **Parameters**
  * **specie** (*str*) – Name of specie.
  * **lattice** (*str*) – The lattice type of structure. e.g. bcc or diamond.
  * **alat** (*float*) – The lattice constant of specific lattice and specie.

### *class* maml.apps.pes._lammps.SpectralNeighborAnalysis(rcutfac, twojmax, element_profile, quadratic=False, \*\*kwargs)

Bases: `LMPStaticCalculator`

Calculator for bispectrum components to characterize the local
neighborhood of each atom in a general way.

Usage:

```none
[(b, db, e)] = sna.calculate([Structure])
b: 2d NumPy array with shape (N, n_bs) containing bispectrum

> coefficients, where N is the No. of atoms in structure and
> n_bs is the No. of bispectrum components.

db: 2d NumPy array with shape (N, 3 \* n_bs \* n_elements)

    containing the first order derivatives of bispectrum
    coefficients with respect to atomic coordinates,
    where n_elements is the No. of elements in element_profile.

e: 2d NumPy array with shape (N, 1) containing the element of

    each atom.
```

#### *CMDS(* = [‘pair_style lj/cut 10’, ‘pair_coeff \* \* 1 1’, ‘compute sna all sna/atom ‘, ‘compute snad all snad/atom ‘, ‘compute snav all snav/atom ‘, ‘dump 1 all custom 1 dump.element element’, ‘dump_modify 1 sort id’, ‘dump 2 all custom 1 dump.sna c_sna[\*]’, ‘dump_modify 2 sort id’, ‘dump 3 all custom 1 dump.snad c_snad[\*]’, ‘dump_modify 3 sort id’, ‘dump 4 all custom 1 dump.snav c_snav[\*]’, ‘dump_modify 4 sort id’_ )

#### \_parse()

Parse results from dump files.

#### \_sanity_check(structure)

Check if the structure is valid for this calculation.

#### \_setup()

Setup a calculation, writing input files, etc.

#### *static* get_bs_subscripts(twojmax)

Method to list the subscripts 2j1, 2j2, 2j of bispectrum
components.

* **Parameters**
  **twojmax** (*int*) – Band limit for bispectrum components.
* **Returns**
  List of all subscripts [2j1, 2j2, 2j].

#### *property* n_bs()

Returns No. of bispectrum components to be calculated.

### *class* maml.apps.pes._lammps.SurfaceEnergy(ff_settings, bulk_structure, miller_indexes, min_slab_size=15, min_vacuum_size=15, lll_reduce=False, center_slab=False, in_unit_planes=False, primitive=True, max_normal_search=None, reorient_lattice=True, box_relax=False, \*\*kwargs)

Bases: `LMPRelaxationCalculator`

Surface energy Calculator.

This calculator generate and calculate surface energies of slab structures based on inputs of
bulk_structure and miller_indexes with the SlabGenerator in pymatgen:
[https://pymatgen.org/pymatgen.core.surface.html](https://pymatgen.org/pymatgen.core.surface.html)

#### calculate()

Calculate surface energies with the formula:
E(Surface) = (E(Slab) - E(bulk)) / Area(surface). (J/m^2).

* **Returns**
  List of miller_indexes with their respective relaxed slab structures and surface energies in J/m^2.

### maml.apps.pes._lammps._pretty_input(lines)

### maml.apps.pes._lammps.*read_dump(file_name, dtype=’float*’)

### maml.apps.pes._lammps.get_default_lmp_exe()

Get lammps executable
Returns: Lammps executable name.

## maml.apps.pes._mtp module

This module provides MTP interatomic potential class.

### *class* maml.apps.pes._mtp.MTPotential(name=None, param=None, version=None)

Bases: `LammpsPotential`

This class implements moment tensor potentials.
Installation of the mlip package is needed.
Please refer to [https://mlip.skoltech.ru](https://mlip.skoltech.ru).

#### *abc_impl(* = <_abc.*abc_data object* )

#### \_line_up(structure, energy, forces, virial_stress)

Convert input structure, energy, forces, virial_stress to
proper configuration format for mlip usage.

* **Parameters**
  * **structure** (*Structure*) – Pymatgen Structure object.
  * **energy** (*float*) – DFT-calculated energy of the system.
  * **forces** (*list*) – The forces should have dimension (num_atoms, 3).
  * **virial_stress** (*list*) – stress should has 6 distinct
    elements arranged in order [xx, yy, zz, yz, xz, xy].

#### evaluate(test_structures, test_energies, test_forces, test_stresses=None, \*\*kwargs)

Evaluate energies, forces and stresses of structures with trained
interatomic potentials.

* **Parameters**
  * **test_structures** (    *[**Structure**]*) – List of Pymatgen Structure Objects.
  * **test_energies** (    *[**float**]*) – List of DFT-calculated total energies of
    each structure in structures list.
  * **test_forces** (    *[**np.array**]*) – List of DFT-calculated (m, 3) forces of
    each structure with m atoms in structures list. m can be varied
    with each single structure case.
  * **test_stresses** (*list*) – List of DFT-calculated (6, ) viriral stresses
    of each structure in structures list.
  * **kwargs** – Parameters of write_param method.

#### *static* from_config(filename, elements)

Initialize potentials with parameters file.

* **Parameters**
  * **filename** (*str*) – The file storing parameters of potentials, filename should
    ends with “.mtp”.
  * **elements** (*list*) – The list of elements.
* **Returns**
  MTPotential

#### pair_coeff(_ = ‘pair_coeff        \* \*_ )

#### pair_style(_ = ‘pair_style        mlip {}_ )

#### read_cfgs(filename)

* **Parameters**
  **filename** (*str*) – The configuration file to be read.

#### train(train_structures, train_energies, train_forces, train_stresses, unfitted_mtp=’08g.mtp’, max_dist=5, radial_basis_size=8, max_iter=1000, energy_weight=1, force_weight=0.01, stress_weight=0.001, init_params=’same’, scale_by_force=0, bfgs_conv_tol=0.001, weighting=’vibration’)

Training data with moment tensor method.

* **Parameters**
  * **train_structures** (    *[**Structure**]*) – The list of Pymatgen Structure object.
    energies ([float]): The list of total energies of each structure
    in structures list.
  * **train_energies** (    *[**float**]*) – List of total energies of each structure in
    structures list.
  * **train_forces** (    *[**np.array**]*) – List of (m, 3) forces array of each structure
    with m atoms in structures list. m can be varied with each single
    structure case.
  * **train_stresses** (*list*) – List of (6, ) virial stresses of each structure
    in structures list.
  * **unfitted_mtp** (*str*) – Define the initial mtp file. Default to the mtp file
    stored in .params directory.
  * **max_dist** (*float*) – The actual radial cutoff.
  * **radial_basis_size** (*int*) – Relevant to number of radial basis function.
  * **max_iter** (*int*) – The number of maximum iteration.
  * **energy_weight** (*float*) – The weight of energy.
  * **force_weight** (*float*) – The weight of forces.
  * **stress_weight** (*float*) – The weight of stresses. Zero-weight can be assigned.
  * **init_params** (*str*) – How to initialize parameters if a potential was not
    pre-fitted. Choose from “same” and “random”.
  * **scale_by_force** (*float*) – Default=0. If >0 then configurations near equilibrium
    (with roughly force < scale_by_force) get more weight.
  * **bfgs_conv_tol** (*float*) – Stop training if error dropped by a factor smaller than this
    over 50 BFGS iterations.
  * **weighting** (*str*) – How to weight configuration with different sizes relative to each other.
    Choose from “vibrations”, “molecules” and “structures”.

#### write_cfg(filename, cfg_pool)

Write configurations to file
:param filename: filename
:type filename: str
:param cfg_pool: list of configurations.
:type cfg_pool: list

Returns:

#### write_ini(mtp_filename=’fitted.mtp’, select=False, \*\*kwargs)

Write mlip.ini file for mlip packages of version mlip-2 or mlip-dev.
Supported keyword arguments are parallel with options stated in the mlip manuals.
mlip-2 is recommended, as it is the only officially supported version by mlip.
Please refer to [https://mlip.skoltech.ru](https://mlip.skoltech.ru).

* **Parameters**
  * **mlip-2** – mtp_filename (str): Name of file with MTP to be loaded.
    write_cfgs (str): Name of file for mlp processed configurations to be written to.
    write_cfgs_skip (int): Skipped number of processed configurations before writing.
    select (bool): activates or deactivates calculation of extrapolation grades and

> optionally writing configurations with high extrapolation grades. False is
> recommended for large-scale MD run.

select_save_selected (str): Name of file for saving configurations with grade

```none
  exceeding select_threshold.
```

select_threshold (float): Configurations with extrapolation grade exceeding the

```none
  value will be saved to the specified file.
```

select_threshold_break (float): The mlip execution will be interrupted if the

```none
  extrapolation grade exceeds this value.
```

select_load_state (str): Name of file for loading the active learning state,

```none
  typically created by the mlp calc-grade command.
```

select_log (str): Name of file (or standard output stream stdout/stderr) for

```none
  writing a log of the configuration selection process.
```

```none
* **mlip-dev** – Abinitio (int): Defines Ab-initio models. Default to 1.
```

> 0: If Ab-initio models is not required.
> 1: Used if driver provides EFS data with configurations.
> 2: Use embedded Lennard-Jones pair potentials.
> > r_min (float): Distance to minimum of pair function (in Angstroms).
> > ```none
> > Default to 2.0.
> > ```
> > scale (float): Value of pair function in minimum (in eV).
> > ```none
> > Default to 1.0.
> > ```
> > cutoff (float): Cutoff radius (in Angstroms). Default to 5.0.
> > 3: Use DFT models by VASP. Linking via files exchange.
> > POSCAR (str): Relative path of POSCAR file.
> > OUTCAR (str): Relative path of OUTCAR file.
> > Start_command (str): Relative path of command file.
> > 4: Use potentials calculating by LAMMPS. Linking via files exchange.
> > Input_file (str): File with configuration to be read by lammps.
> > Output_file (str): File with configuration and EFS data to be read by MLIP.
> > Start_command (str): Relative path of command file.
> > 5: Use MTP as Ab-initio potentials.
> > MTP_filename (str): MTP file name.

MLIP (str): MTP.

> load_from (str): Potential filename.
> Cacluate_EFS (bool): Whether to perform EFS calculation by MTP.
> Fit (bool): Whether to perform MTP learning.
> > Save (str): Output MTP file name (for trained MTP).
> > Energy_equation_weight (float): Weight for energy equation in
> > > fitting procedure. Default to 1.0.
> > > Forces_equation_weight (float): Weight for forces equations in
> > ```none
> > fitting procedure. Default to 0.001.
> > ```
> > Stress_equation_weight (float): Weight for stresses equations in
> > ```none
> > fitting procedure.  Default to 0.1.
> > ```
> > Relative_forces_weight (float): If greater than zero, large forces
> > ```none
> > will be fitted less accurate than small. Default to 0.0.
> > ```
> > Fit_log (str): File to write fitting log. No logging if not specified.
> > ```none
> > Default to None.
> > ```
> Select (bool): Whether to activate active learning. Default to False.
> > Site_E_weight (float): Weight for site energy equations in
> > ```none
> > selection procedure. Default to 1.0.
> > ```
> > Energy_weight (float): Weight for energy equation in
> > ```none
> > selection procedure. Default to 0.0.
> > ```
> > Forces_weight (float): Weight for forces equations in
> > ```none
> > selection procedure. Default to 0.0.
> > ```
> > Stress_weight (float): Weight for stresses equations in
> > ```none
> > selection procedure. Default to 0.0.
> > ```
> > Threshold_slct (float): Selection threshold - maximum
> > ```none
> > allowed extrapolation level. Default to 0.1.
> > ```
> > Save_TS (str): Filename where selected configurations
> > ```none
> > will be saved. No configuration saving if not specified.
> > Default to None.
> > ```
> > Save_state (str): Filename where state of the selection
> > ```none
> > will be saved. No saving if not specified. Default to None.
> > ```
> > Load_state (str): Filename where state of the selection
> > ```none
> > will be loaded. No saving if not specified. Default to None.
> > ```
> > Select_log (str): File to write fitting log. No logging
> > ```none
> > if not specified. Default to None.
> > ```
> LOFT (bool): Whether to perform learning on the fly. Default to False
> > EFSviaMTP (bool): Works only on LOFT regime. If True,
> > ```none
> > only MTP-calculated EFS will be passed to driver, else
> > pass to driver ab-initio EFS while LOTF when learning occurs.
> > ```
> > Log (str): Filename to write log of learning on the fly process.
> > ```none
> > No logging if not specified. Default to None.
> > ```
> Check_errors (bool): If True, comparison and accumulation of
> ```none
> error statistics for EFS calculated by ab-initio models and MTP.
> Default to False.
> ```
> ```none
> Log (str): Filename to write log of learning on the fly process.
> ```
> ```none
>     No logging if not specified. Default to None.
> ```
> Write_cfgs (bool): File for writing all processed configurations.
> ```none
> No configuration recording if not specified. Default to None.
> ```
> ```none
> Skip_N (int): The number of configurations to skip while writing.
> ```
> ```none
>     Default to 0.
> ```
> Log (str): Filename to write MLIP log. No logging if not specified.
> ```none
> Default to None.
> ```

Driver (int): Defines the configuration driver. Default to 1.

> 0: No driver or external MD driver.
> 1: Read configurations from database file.
> > Database_filename (str): Configuration file name.
> > Max_count (int): Maximal number of configurations to read.
> > Log (str): Filename to write reading log. No logging
> > > if not specified. Default to None.
> > > 2: Embedded algorithm for relaxation.
> > > Pressure (float): External pressure (in GPa).
> > ```none
> > If not zero enthalpy is minimized. Default to 0.0.
> > ```
> > Iteration_limit (int): Maximal number of iteration of
> > ```none
> > the relaxation process. Default to 500.
> > ```
> > Min_dist (float): Minimal interatomic distance constraint
> > ```none
> > (in Angstroms). Default to 1.0.
> > ```
> > Forces_tolerance (float): Forces on atoms in relaxed
> > ```none
> > configuration should be smaller than this value
> > (in eV/Angstroms). Default to 0.0001.
> > ```
> > Stress_tolerance (float): Stresses in relaxed configuration
> > ```none
> > should be smaller than this value (in GPa). Default to 0.001.
> > ```
> > Max_step (float): Maximal allowed displacement of atoms and
> > ```none
> > lattice vectors in Cartesian coordinates (in Angstroms).
> > Default to 0.5.
> > ```
> > Min_step (float): Minimal displacement of atoms and
> > ```none
> > lattice vectors in Cartesian coordinates (in Angstroms).
> > Default to 1.0e-8.
> > ```
> > BFGS_Wolfe_C1 (float): Wolfe condition constant on the function
> > ```none
> > decrease (linesearch stopping criterea). Default to 1.0e-3.
> > ```
> > BFGS_Wolfe_C2 (float): Wolfe condition constant on the gradient
> > ```none
> > decrease (linesearch stopping criterea). Default to 0.7.
> > ```
> > Save_relaxed (str): Filename for output results of relaxation.
> > ```none
> > No configuration will be saved if not specified.
> > Default to None.
> > ```
> > Log (str): Filename to write relaxation log. No logging
> > ```none
> > if not specified. Default to None.
> > ```

#### write_param(fitted_mtp=’fitted.mtp’, \*\*kwargs)

Write fitted mtp parameter file to perform lammps calculation.

* **Parameters**
  **fitted_mtp** (*str*) – Filename to store xml formatted parameters.

### maml.apps.pes._mtp.feed(attribute, kwargs, dictionary, tab=’\\t’)

* **Parameters**
  * **attribute** (*str*) – Attribute to be operated.
  * **kwargs** (*dict*) – Generic parameters.
  * **dictionary** (*dict*) – The default parameters dictionary.
  * **tab** (*str*) – ‘t’ or ‘tt’, depend on orders of attribute.
* **Returns**
  (str).

## maml.apps.pes._nnp module

This module provides NNP interatomic potential class.

### *class* maml.apps.pes._nnp.NNPotential(name=None, param=None, weight_param=None, scaling_param=None)

Bases: `LammpsPotential`

This class implements Neural Network Potential.

#### *abc_impl(* = <_abc.*abc_data object* )

#### \_line_up(structure, energy, forces, virial_stress)

Convert input structure, energy, forces, virial_stress to
proper configuration format for n2p2 usage. Note that
n2p2 takes bohr as length unit and Hartree as energy unit.

* **Parameters**
  * **structure** (*Structure*) – Pymatgen Structure object.
  * **energy** (*float*) – DFT-calculated energy of the system.
  * **forces** (*list*) – The forces should have dimension
    (num_atoms, 3).
  * **virial_stress** (*list*) – stress should has 6 distinct
    elements arranged in order [xx, yy, zz, xy, yz, xz].

#### bohr_to_angstrom(_ = 0.52917721090_ )

#### eV_to_Ha(_ = 0.03674932217565_ )

#### evaluate(test_structures, test_energies, test_forces, test_stresses=None)

Evaluate energies, forces and stresses of structures with trained
interatomic potentials.

* **Parameters**
  * **test_structures** (    *[**Structure**]*) – List of Pymatgen Structure Objects.
  * **test_energies** (    *[**float**]*) – List of DFT-calculated total energies of
    each structure in structures list.
  * **test_forces** (    *[**np.array**]*) – List of DFT-calculated (m, 3) forces of
    each structure with m atoms in structures list. m can be varied
    with each single structure case.
  * **test_stresses** (*list*) – List of DFT-calculated (6, ) viriral stresses
    of each structure in structures list.

#### *static* from_config(input_filename, scaling_filename, weights_filenames)

Initialize potentials with parameters file.

* **Parameters**
  * **input_filename** (*str*) – The file storing the input configuration of
    Neural Network Potential.
  * **scaling_filename** (*str*) – The file storing scaling info of
    Neural Network Potential.
  * **weights_filenames** (*list*) – List of files storing weights of each specie in
    Neural Network Potential.

#### load_input(filename=’input.nn’)

Load input file from trained Neural Network Potential.

* **Parameters**
  **filename** (*str*) – The input filename.

#### load_scaler(scaling_filename)

Load scaling info of trained Neural Network Potential.

* **Parameters**
  **scaling_filename** (*str*) – The scaling file.

#### load_weights(weights_filename, specie)

Load weights file of trained Neural Network Potential.

* **Parameters**
  * **weights_filename** (*str*) – The weights file.
  * **specie** (*str*) – The name of specie.

#### pair_coeff(_ = ‘pair_coeff        \* \* {}_ )

#### pair_style(_ = ‘pair_style        nnp dir “./” showew no showewsum 0 maxew 10000000 resetew yes cflength 1.8897261328 cfenergy 0.0367493254_ )

#### read_cfgs(filename=’output.data’)

Read the configuration file.

* **Parameters**
  **filename** (*str*) – The configuration file to be read.

#### train(train_structures, train_energies, train_forces, train_stresses=None, \*\*kwargs)

Training data with moment tensor method.

* **Parameters**
  * **train_structures** (    *[**Structure**]*) – The list of Pymatgen Structure object.
    energies ([float]): The list of total energies of each structure
    in structures list.
  * **train_energies** (    *[**float**]*) – List of total energies of each structure in
    structures list.
  * **train_forces** (    *[**np.array**]*) – List of (m, 3) forces array of each structure
    with m atoms in structures list. m can be varied with each
    single structure case.
  * **train_stresses** (*list*) – List of (6, ) virial stresses of each
    structure in structures list.
  * **kwargs** – Parameters in write_input method.

#### write_cfgs(filename, cfg_pool)

Write the formatted configuration file.

* **Parameters**
  * **filename** (*str*) – The filename to be written.
  * **cfg_pool** (*list*) – The configuration pool contains
    structure and energy/forces properties.

#### write_input(\*\*kwargs)

Write input.nn file to train the Neural Network Potential.

* **Parameters**
  * **atom_energy** (*float*) – Atomic reference energy.
  * **kwargs** – General nnp settings:

atom_energy (dict): Free atom reference energy for each specie.
cutoff_type (int): Type of cutoff function. Default to 1

> (i.e., cosine function).

scale_features (int): Determine the method to scale the

```none
  symmetry function.
  0: no scaling.
  1: scale_symmetry_functions.
  2: center_symmetry_functions.
  3. scale_symmetry_functions_sigma.
```

scale_min_short (float): Minimum value for scaling.

```none
  Default to 0.0.
```

scale_max_short (float): Maximum value for scaling.

```none
  Default to 1.
```

hidden_layers (list): List of the numbers of

```none
  nodes in each hidden layer.
```

activations (str): Activation function for each hidden layer.

```none
  ’t’: tanh, ‘s’: logistic, ‘p’: softplus.
```

normalize_nodes (boolean): Whether to normalize input of nodes.

Additional settings for training:

```none
  epoch (int): Number of training epochs.
  updater_type (int): Weight update method

  > 0: gradient Descent, 1: Kalman filter.

  parallel_mode (int): Training parallelization used.

      Default to serial mode.

  jacobian_mode (int): Jacobian computation mode.

      0: Summation to single gradient,
      1: Per-task summed gradient,
      2: Full Jacobian.

  update_strategy (int): Update strategy.

      0: combined, 1: per-element.

  selection_mode (int): Update candidate selection mode.

      0: random, 1: sort, 2: threshold

  task_batch_size_energy (int): Number of energy update

      candidates prepared per task for each update.

  task_batch_size_force (int): Number of force update

      candidates prepared per task for each update.

  test_fraction (float): Fraction of structures kept for

      testing.

  force_weight (float): Weight of force updates relative

      to energy updates. Default to 10.0

  short_energy_fraction (float): Fraction of energy updates

      per epoch. Default to 1.0.

  short_force_fraction (float): Fraction of force updates

      per epoch. Default to 0.02315.

  short_energy_error_threshold (float): RMSE threshold for

      energy update candidates. Default to 0.0.

  short_force_error_threshold (float): RMSE threshold for

      force update candidates. Default to 1.0.

  rmse_threshold_trials (int): Maximum number of RMSE

      threshold trials. Default to 3.

  weights_min (float): Minimum value for initial random

      weights. Default to -1.

  weights_max (float): Maximum value for initial random

      weights. Default to 1.

  write_trainpoints (int): Write energy comparison every

      this many epochs. Default to 1.

  write_trainforces (int): Write force comparison every

      this many epochs. Default to 1.

  write_weights_epoch (int): Write weights every this many

      epochs. Default to 1.

  write_neuronstats (int): Write neuron statistics every

      this many epochs. Default to 1.

  # Kalman Filter
  kalman_type (int): Kalman filter type. Default to 0.
  kalman_epsilon (float): General Kalman filter parameter

  > epsilon. Default to 0.01.

  kalman_q0 (float): General Kalman filter parameter q0.

      Default to 0.01.

  kalman_qtau (float): General Kalman filter parameter

      qtau. Default to 2.302.

  kalman_qmin (float): General Kalman filter parameter qmin.

      Default to 1e-6.

  kalman_eta (float): Standard Kalman filter parameter eta.

      Default to 0.01.

  kalman_etatau (float): Standard Kalman filter parameter

      etatau. Defaults to 2.302.

  kalman_etamax (float): Standard Kalman filter parameter

      etamax. Default to 1.0.
```

Symmetry functions:

```none
  r_cut (float): Cutoff distance (unit: Å).
  r_etas (numpy.array): η in radial function.
  r_shift (numpy.array): Rs in radial function.
  a_etas (numpy.array): η in angular function.
  zetas (numpy.array): ζ in angular function.
  lambdas (numpy.array): λ in angular function. Default to (1, -1).
```

#### write_param()

Write optimized weights file to perform energy and force prediction.

## maml.apps.pes._snap module

This module provides SNAP interatomic potential class.

### *class* maml.apps.pes._snap.SNAPotential(model, name=None)

Bases: `LammpsPotential`

This class implements Spectral Neighbor Analysis Potential.

#### *abc_impl(* = <_abc.*abc_data object* )

#### evaluate(test_structures, test_energies, test_forces, test_stresses=None, include_stress=False, stress_format=’VASP’)

Evaluate energies, forces and stresses of structures with trained
machinea learning potentials.

* **Parameters**
  * **test_structures** (    *[**Structure**]*) – List of Pymatgen Structure Objects.
  * **test_energies** (    *[**float**]*) – List of DFT-calculated total energies of
    each structure in structures list.
  * **test_forces** (    *[**np.array**]*) – List of DFT-calculated (m, 3) forces of
    each structure with m atoms in structures list. m can be varied
    with each single structure case.
  * **test_stresses** (*list*) – List of DFT-calculated (6, ) viriral stresses
    of each structure in structures list.
  * **include_stress** (*bool*) – Whether to include stress components.
  * **stress_format** (*str*) – stress format, default to “VASP”

#### *static* from_config(param_file, coeff_file, \*\*kwargs)

Initialize potentials with parameters file and coefficient file.

* **Parameters**
  * **param_file** (*str*) – The file storing the configuration of potentials.
  * **coeff_file** (*str*) – The file storing the coefficients of potentials.
* **Returns**
  SNAPotential.

#### pair_coeff(_ = ‘pair_coeff        \* \* {coeff_file} {param_file} {elements}_ )

#### pair_style(_ = ‘pair_style        snap_ )

#### train(train_structures, train_energies, train_forces, train_stresses=None, include_stress=False, stress_format=’VASP’, \*\*kwargs)

Training data with models.

* **Parameters**
  * **train_structures** (    *[**Structure**]*) – The list of Pymatgen Structure object.
    energies ([float]): The list of total energies of each structure
    in structures list.
  * **train_energies** (    *[**float**]*) – List of total energies of each structure in
    structures list.
  * **train_forces** (    *[**np.array**]*) – List of (m, 3) forces array of each
    structure with m atoms in structures list. m can be varied with
    each single structure case.
  * **train_stresses** (*list*) – List of (6, ) virial stresses of each
    structure in structures list.
  * **include_stress** (*bool*) – Whether to include stress components.
  * **stress_format** (*string*) – stress format, default to VASP

#### write_param()

Write parameter and coefficient file to perform lammps calculation.