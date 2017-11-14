import sys
sys.dont_write_bytecode = True
import pytest
import numpy as np
import os
import murnaghan2017 as m

def test_preprocess_file_bad():
    """
    entering bad name for energy driver raises value error
    """
    with pytest.raises(ValueError):
        m.preprocess_file(0, 0, 'dummyfilename', 'asdfas')

def test_preprocess_file_abinit():
    """
    calling _preprocess_file with for abinit correctly writes lattice constants and angles to abinit input file.
    
    not very robust since different number format would cause a fail
    """
    template_file = 'abinit.in.template.example'
    # preprocess
    s = [0.1, 0.2, 0.3]
    ang = [90, 90, 120]
    m.preprocess_file(s, ang, template_file, 'abinit')
    # compare writetn abinit file with correct input file
    with open('abinit.in.correct') as f1, open('abinit.in') as f2:
        assert f1.readlines() == f2.readlines()
    os.remove('abinit.in') 

def test_generate_lattice_constants():
    """ 
    generates numpy array of lattice paramters correctly
    """
    abc_guess = [0.5, 1, 2.0]
    pert = 0.05
    N = 5
    abc = m.generate_lattice_constants(abc_guess, pert, N)
    abc_correct = np.array([[0.475, 0.95, 1.9], [0.4875, 0.975, 1.95],
        [0.5, 1., 2.], [0.5125, 1.025, 2.05], [0.525, 1.05, 2.1]])
    assert (abc == abc_correct).all()

def test_generate_lattice_constants_2d():
    """ 
    generates numpy array of lattice paramters correctly
    """
    abc_guess = [0.5, 1, 2.0]
    pert = 0.05
    N = 5
    abc = m.generate_lattice_constants(abc_guess, pert, N, two_dim=True)
    abc_correct = np.array([[0.475, 0.95, 2], [0.4875, 0.975, 2],
        [0.5, 1., 2.], [0.5125, 1.025, 2], [0.525, 1.05, 2]])
    assert (abc == abc_correct).all()
    
def test_run_energy_calculations_abinit():
    """
    sets up files and runs abinit for each lattice constant
    
    Requires abinit set up correctly
    """
    abc = np.array([[a*pert for a in [4,4,6]] for pert in [0.95, 1, 1.05]])
    print abc
    angles=[90, 90, 90]
    energy_driver='abinit'
    m.run_energy_calculations(abc, angles, energy_driver)
    assert False

def test_abinit_get_energy():
    assert False

def test_run_abinit():
    assert False

def test_fit_to_murnaghan():
    assert False

