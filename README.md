# murnaghan2017
Fit Murnaghan equation of state to energy/volume data. Also, easily calculate energy/volume data using some DFT codes.

Currently, this is only set up for the Abinit and Socorro DFT codes, though others could be added easily.

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

## General setup
You will need to run a python script which sets up the lattice parameter sweep and does the post processing. The example script can be modified for your system.
Assumes no unit cell relaxation happens during a single call to the dft code.

## Setup for Socorro
* Create a directory called templatedir/
* Put input files *crystal* and *argvf* and pseudopotentials in templatedir/
* Rename *crystal* file to *crystal.template*
* **Delete scale and primitve vector lines from crystal file**
* I like to put the pseudopotentials in the main directory and symlink up to them to save space

## Setup for abinit
* Create a directory called templatedir/
* Put all abinit input files in templatedir/ (usually just a files file and input file, and possibly pseudopotentials)
* The files file should be named *files* and the input file should be named *abinit.in.template*
* Set the first line of *files* file to "abinit.in", which is the name of the input files these scripts will create
* **All keywords and values involving unit cell definition in _abinit.in.template_ should be commented or deleted. These include _acell_, _rprim_, _angdeg_, _scalecart_, _brvltt_, and _spgroup_.**

## Results
* *energies.dat* contains raw data
* *murnaghan_paramters.dat* contains fitted murnaghan parameters
* *plot_murnaghan.py* will plot the fit and raw data vs both volume and scale *a*. It can be modified to plot against other lattice vector scale values if needed. 
* "invalid error encountered in power" error usually means raw energy data is bad. You can read energies.dat or visualize with plot_murnaghan.py
