# murnaghan2017
Calculate energy vs. lattice parameter data using some DFT codes with an easy to use Python interface. Also, fit Murnaghan equation of state to energy vs. volume data, or a bivariate cubic polynomial to energy data on a grid of two lattice parameters (useful, for example, in hexagonal structures).

Currently, this is set up for the Abinit, Socorro, and Elk DFT codes, though others could be added easily.

If pytest won't work, or the modules can't be found, you can try loading an anaconda module. I've had issues with older versions of pytest

Tested with: python 2.7.12, 2.7.13, 2.7.5, pytest 3.0.7, abinit 8.0.8, elk 2.3.22, 4.3.6

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
or if the above pytest commands don't work
```bash
$ python -m pytest tests/test_murnaghan2017.py
```
If you don't have Abinit, Socorro, or Elk set up on your computer, then the corresponding integration tests will fail.

## General setup
You will need to create a python script which imports the murnaghan2017 module, sets up the lattice parameter sweep, and does the post processing. 
This repository includes an example script, *example_run.py*, which you can start with and modify for your system. 
The dft code to use and the template file are specified in this script. 
You will also need to specify the unscaled primitive vectors for your unit cell, and the lattice vector scales to sweep. 
The lattice vector scales get multiplied by the unscaled primitive vectors.

For example, for the Nth lattice parameters tested, the lattice vectors will be
```latex
abc_list[N,i]*pvu[i,:] for i=0,1,2
```
with lattice parameters
```latex
abc_list[N,i]*|pvu[i,:]| for i=0,1,2
```

The post processing assumes no unit cell relaxation happens during a single call to the dft code. Relaxation of atomic positions is okay in theory, but could cause problems because the functions to read the total energy from the DFT code output havn't been tested on output files from structure relaxations.

### Alternative unit cell setup using lattice vector angles
As an alternative to specifying the unscaled lattice vectors, *prim_vec_unscaled*, you can instead specify the angles between vectors, *angles*. This will internally create unscaled primitive vectors of unit length which are then scaled as above by *abc_list*. For example:
```python
volumes, energy_list_hartree = murnaghan2017.lattice_parameter_sweep(energy_driver, 
   template_file, abc_list, angles=[90, 90, 120])
```
instead of
```python
volumes, energy_list_hartree = murnaghan2017.lattice_parameter_sweep(energy_driver, 
   template_file, abc_list, prim_vec_unscaled=[[1,0,0],[cos(120),sin(120),0],[0,0,1]])
```
### More details about lattice parameter sweeps
For more detailed information on setting up the lattice parameter sweep, see the docstring for murnaghan2017.lattice_parameter_sweep. 

### Fitting energy data and finding equilibrium lattice parameters
See *example_run.py* for examples using the MurnaghanFit and Poly2DFit classes to fit the energy data and output equilibrium lattice parameters. The Murnaghan fit is useful for materials with only one lattice parameter. The bivariate polynomial fit is useful for materials with two independent lattice parameters, such as hexagonal and wurtzite.

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

## Setup for Quantum Espresso
* Create a directory called templatedir/
* Put all Quantum Espresso input files in templatedir/ (usually just one input file and possibly pseudopotentials)
* Rename the main input file to *espresso.in.template*
* **Delete CELL PARAMETERS section in _espresso.in.template_**.

## Setup for Elk
* Create a directory called templatedir/
* Put your elk input file *elk.in* in templatedir/
* rename *elk.in* to *elk.in.template*
* **All keywords and values involving unit cell definition in _elk.in.template_ should be commented or deleted. These include _scale_, _scale1_, _scale2_, _scale3_, and _avec_.**

## Run lattice parameter sweep and fit
Using the included example script,
```bash
$ pytest example_run.py
```
This should run N instances of the dft code in labeled work directories. The calculated energies are written to *energies.dat* and the fitted murnaghan parameters including lattice constant and bulk modulus are written to *murnaghan_parameters.dat*.

## Results
* *energies.dat* contains raw energy vs. lattice constant and volume data
* *murnaghan_parameters.dat* contains fitted murnaghan parameters (if a murnaghan fit is performed)
* *plot_murnaghan.py* will plot the fit and raw data vs both volume and scale *a* (if a murnaghan fit is performed). It can be modified to plot against other lattice vector scale values if needed. 
* *poly2d_parameters.dat* contains fitted polynomial parameters (if a bivariate polynomial fit is performed)
* *plot_poly2d.py* will plot the fit and raw data vs *a* and *c* (if a bivariate polynomial fit is performed). It can be modified to plot against other lattice vector scale values if needed. 
* Be sure to carefully examine the input files and output files in the work directories, as well as the final raw data to make sure everything was set up and run correctly.


## Notes
* "invalid error encountered in power" error usually means raw energy data is bad. You can read energies.dat or visualize with plot_murnaghan.py
* The integration tests call and run their respective dft codes. Thus an integration test will fail if that code is not installed, which is okay if you don't plan to use that code. 
* The integration tests compare energy and other results from a dft calculation to references results I generated. Your results could be slightly different than mine because of small differences between versions and installations of the code. Because of this, I set some hopefully reasonable tolerances for comparing these results. If an integration test fails because of an assertion error, it could be that the tolerances are still too strict.
