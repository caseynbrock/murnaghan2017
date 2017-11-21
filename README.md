## murnaghan2017
Fit Murnaghan equation of state to energy/volume data. Also, easily calculate energy/volume data using some DFT codes.

Currently, this is only set up for the Abinit and Socorro DFT codes, though others could be added easily.

# Running tests
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

# General setup
You will need to run a python script which sets up the lattice parameter sweep and does the post processing. The example script can be modified for your system.
Assumes no unit cell relaxation happens during a single call to the dft code.

# Setup for Socorro
* Create a directory called templatedir/
* Put input files *crystal* and *argvf* and pseudopotentials in templatedir/
* **Delete scale and primitve vector lines from crystal file**

# Setup for abinit
* Create a directory called templatedir/
* Put all abinit input files in templatedir/ (usually just a files file and input file, and possibly pseudopotentials)
* The files file should be named *files* and the input file should be named *abinit.in.template*
* Set the first line of *files* file to "abinit.in", which is the name of the input files these scripts will create
* **All keywords and values involving unit cell definition in _abinit.in.template_ should be commented or deleted. These include _acell_, _rprim_, _angdeg_, _scalecart_, _brvltt_, and _spgroup_.**
