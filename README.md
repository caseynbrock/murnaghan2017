# murnaghan2017
Fit Murnaghan equation of state to energy/volume data. Also, easily calculate energy/volume data using some DFT codes.

Currently, this is only set up for the Abinit DFT code.

To get running:
) clone repository
) run tests:
$ pytest 
or my preference
$ pytest -v
or to exclude integration tests (which actually run the dft codes)
$ pytest -k "not integration"
or to exclude tests for a certain dft code (abinit for example)
$ pytest -k "not abinit"
or to run a specific test
$ pytest tests/test_murnaghan2017.py::test_preprocess_file_abinit_rprim


Setup for abinit:
this directory should contain abinit.in.template, which gets edited by the murnaghan2017 script.
It should also contain a directory called templatedir containing the abinit files file, which should be named 'files'. 
This templatedir gets copied to a new working directory, workdir.<i>, for each abinit run.
If the abinit runs require any more input files, they should be placed in templatedir.
Note that these scripts name the abinit input file abinit.in, so that should be the first line of the files file.
