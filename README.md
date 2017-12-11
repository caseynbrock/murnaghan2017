# murnaghan2017
Fit Murnaghan equation of state to energy/volume data. Also, easily calculate energy/volume data using some DFT codes.

Currently, this is only set up for the Abinit, Socorro, and Elk DFT codes, though others could be added easily.

If pytest won't work, or the modules can't be found, you can try loading an anaconda module. I've had issues with older versions of pytest

Tested with: python 2.7.12, 2.7.13, 2.7.5, pytest 3.0.7, abinit 8.0.8

## Running tests
In the main directory, run
```bash
$ pytest
```
or my preference
```bash
$ pytest -v
```
or to exclude integration tests (which actually run the dft codes)
```bash
$ pytest -k "not integration"
```
or to exclude tests for a certain dft code (abinit for example)
```bash
$ pytest -k "not abinit"
```
or to run a specific test
```bash
$ pytest tests/test_murnaghan2017.py::test_preprocess_file_abinit_rprim
```

If you don't have Abinit, Socorro, or Elk set up on your computer, then the corresponding integration tests will fail.

## General setup
You will need to create a python script which imports the murnaghan2017 module, sets up the lattice parameter sweep, and does the post processing. This repository includes an example script, *example_run.py*, which you can start with and modify for your system. The dft code to use and the template file are specified in this script. You will also need to specify the unscaled primitive vectors for your unit cell, guesses for lattice parameters (these get multiplied by the unscaled primitive vectors), and a list of scales at least 4 elements long.

For example, for the Nth scale tested, the lattice vectors will be
```latex
s[N]*abc_guess[i]*pvu[i] for i=0,1,2
```

The post processing assumes no unit cell relaxation happens during a single call to the dft code (atomic position relaxation is okay).

### Notes for two-dimensional materials
For two-dimensional materials, set two_dim=True when instantiating the LatticeParameterSweep object. This option tells the code to scale only the *a* and *b* lattice parameters while leaving *c* fixed. For example:
```python
sweep = m.LatticeParameterSweep(ed, tf, s, abcg, prim_vec_unscaled=pvu, two_dim=True)
```

### Alternative unit cell setup using lattice vector angles
As an alternative to specifying the unscaled lattice vectors, *prim_vec_unscaled*, you can instead specify the angles between vectors, *angles*. This will internally create unscaled primitive vectors of unit length which are then scaled as above by *abc_guess* and *scale*. For example:
```python
sweep = m.LatticeParameterSweep(ed, tf, s, abcg, angles=[90, 90, 120])
```

## Setup for Socorro
* Create a directory called templatedir/
* Put input files *crystal* and *argvf* and pseudopotentials in templatedir/
* Rename *crystal* file to *crystal.template*
* **Delete scale and primitve vector lines from crystal file**
* I like to put the pseudopotentials in the main directory and symlink up to them to save space

## Setup for Abinit
* Create a directory called templatedir/
* Put all abinit input files in templatedir/ (usually just a files file and input file, and possibly pseudopotentials)
* The files file should be named *files* and the input file should be named *abinit.in.template*
* Set the first line of *files* file to "abinit.in", which is the name of the input files these scripts will create
* **All keywords and values involving unit cell definition in _abinit.in.template_ should be commented or deleted. These include _acell_, _rprim_, _angdeg_, _scalecart_, _brvltt_, and _spgroup_.**

## Setup for Elk
* Create a directory called templatedir/
* Put your elk input file *elk.in* in templatedir/
* rename *elk.in* to *elk.in.template*
* **All keywords and values involving unit cell definition in _elk.in.template_ should be commented or deleted. These include _scale_, _scale1_, _scale2_, _scale3_, and _avec_.**

## Run lattice parameter sweep
Using the included example script,
```bash
$ pytest example_run.py
```
This should run N instances of the dft code in labeled work directories. The calculated energies are written to *energies.dat* and the fitted murnaghan parameters including lattice constant and bulk modulus are written to *murnaghan_parameters.dat*.

## Results
* *energies.dat* contains raw data
* *murnaghan_parameters.dat* contains fitted murnaghan parameters
* *plot_murnaghan.py* will plot the fit and raw data vs both volume and scale *a*. It can be modified to plot against other lattice vector scale values if needed. 
* Be sure to carefully examine the input files and output files in the work directories, as well as the final raw data to make sure everything was set up and run correctly.

## Notes
* "invalid error encountered in power" error usually means raw energy data is bad. You can read energies.dat or visualize with plot_murnaghan.py
* The integration tests call and run their respective dft codes. Thus an integration test will fail if that code is not installed, which is okay if you don't plan to use that code. 
* The integration tests compare energy and other results from a dft calculation to references results I generated. Your results could be slightly different than mine because of small differences between versions and installations of the code. Because of this, I set some hopefully reasonable tolerances for comparing these results. If an integration test fails because of an assertion error, it could be that the tolerances are still too strict.
