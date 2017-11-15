import sys
sys.dont_write_bytecode = True
import pytest
import numpy as np
import os
import shutil
from contextlib import contextmanager
import tempfile
import murnaghan2017 as m

main_dir = os.getcwd()
test_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = os.path.join(test_dir, 'input_files')

class TemporaryDirectory(object):
    """
    Context manager for tempfile.mkdtemp() so it's usable with 'with' statement.
    This version automatically changes to and leaves temporary directory.
    """
    def __enter__(self):
        self.start_dir = os.getcwd()
        self.name = tempfile.mkdtemp()
        os.chdir(self.name)
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.start_dir)
        shutil.rmtree(self.name)

def test_angles_prim_vec_neither():
    """
    raises value error if neither of angles and prim_vec are specified
    """
    energy_driver = None
    template_file = None
    abc = None
    with pytest.raises(ValueError):
        sweep = m.LatticeParameterSweep(energy_driver, template_file, abc)

def test_angles_prim_vec_both():
    """
    raises value error if both angles and prim_vec are specified
    """
    energy_driver = None
    template_file = None
    abc = None
    with pytest.raises(ValueError):
        sweep = m.LatticeParameterSweep(energy_driver, template_file, abc, angles=1, prim_vec=1)

def test_angles_None_when_prim_vec_specified():
    """
    if prim_vec is specified, self.angles=None
    """
    energy_driver = None
    template_file = None
    abc = None
    sweep = m.LatticeParameterSweep(energy_driver, template_file, abc, prim_vec=1)
    assert sweep.angles is None

def test_preprocess_file_bad():
    """
    entering bad name for energy driver raises value error
    """
    energy_driver = 'dummy'
    s_dummy = [1,2,3]
    sweep = m.LatticeParameterSweep(energy_driver, '', [], prim_vec=[])
    with pytest.raises(ValueError):
        sweep.preprocess_file(s_dummy)

def test_preprocess_file_abinit_angdeg():
    """
    calling _preprocess_file with for abinit correctly writes lattice constants and angles to abinit input file.
    
    not very robust since different number format would cause a fail
    """
    energy_driver = 'abinit'
    template_file = os.path.join(input_dir, 'abinit.in.template.example')
    correct_file = os.path.join(input_dir, 'abinit.in.correct')
    s = [0.1, 0.2, 0.3]
    ang = [90, 90, 120]
    sweep = m.LatticeParameterSweep(energy_driver, template_file, s, angles=ang)
    sweep.preprocess_file(s)
    # compare written abinit file with correct input file
    with open(correct_file) as f1, open('abinit.in') as f2:
        assert f1.readlines() == f2.readlines()
    os.remove('abinit.in') 

def test_preprocess_file_abinit_angdeg2():
    """
    calling _preprocess_file with for abinit correctly writes lattice constants and angles to abinit input file.
    
    not very robust since different number format would cause a fail
    """
    with TemporaryDirectory() as tmp_dir:
        energy_driver = 'abinit'
        template_file = os.path.join(input_dir, 'abinit.in.template.example')
        correct_file = os.path.join(input_dir, 'abinit.in.correct')
        s = [0.1, 0.2, 0.3]
        ang = [90, 90, 120]
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, angles=ang)
        sweep.preprocess_file(s)
        # compare written abinit file with correct input file
        with open(correct_file) as f1, open('abinit.in') as f2:
            assert f1.readlines() == f2.readlines()

def test_preprocess_file_abinit_rprim():
    """
    calling _preprocess_file with for abinit correctly writes lattice constants and angles to abinit input file.
    
    not very robust since different number format would cause a fail
    """
    energy_driver = 'abinit'
    template_file = os.path.join(input_dir, 'abinit.in.template.example')
    correct_file = os.path.join(input_dir, 'abinit.in.correct.2')
    s = [0.1, 0.2, 0.3]
    prim_vec = [[1,1,0], [0,2,2], [3,0,3]]
    sweep = m.LatticeParameterSweep(energy_driver, template_file, s, prim_vec=prim_vec)
    sweep.preprocess_file(s)
    # compare written abinit file with correct input file
    with open(correct_file) as f1, open('abinit.in') as f2:
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
    

def test_prim_vec_from_angles_hex():
    energy_driver='dummy'
    template_file = 'dummy'
    abc = []
    angles=[90, 90, 120]
    sweep = m.LatticeParameterSweep(energy_driver, template_file, abc, angles=angles)
    correct_vec = np.array([[1,0,0],[-0.5, np.sqrt(3)/2, 0], [0,0,1]])
    assert np.isclose(sweep._prim_vec_from_angles(), correct_vec).all()

def test_prim_vec_from_angles_tet():
    energy_driver='dummy'
    template_file = 'dummy'
    abc = []
    angles=[90, 90, 90]
    sweep = m.LatticeParameterSweep(energy_driver, template_file, abc, angles=angles)
    correct_vec = np.array([[1,0,0],[0,1,0], [0,0,1]])
    assert np.isclose(sweep._prim_vec_from_angles(), correct_vec).all()

def test_get_energy_bad():
    """
    entering bad name for energy driver raises value error
    """
    energy_driver = 'dummy'
    s_dummy = [1,2,3]
    sweep = m.LatticeParameterSweep(energy_driver, '', [], prim_vec=[])
    with pytest.raises(ValueError):
        sweep.get_energy()

def test_abinit_get_energy_single():
    """When only single etotal reported (typical case), returns etotal in Hartree"""
    energy_driver='abinit'
    template_file = None
    abc = None
    prim_vec = 1
    sweep = m.LatticeParameterSweep(energy_driver, template_file, abc, prim_vec=prim_vec)
    os.chdir(test_dir) 
    shutil.copy(os.path.join(input_dir, 'log.example'), 'log')
    E = sweep.get_energy()
    os.remove('log')
    os.chdir('..')
    assert np.isclose(E, -6.1395717098)
    

def test_abinit_get_energy_multiple():
    """When multiple etotal reported (as in structure relaxation), returns FINAL etotal in Hartree"""
    energy_driver='abinit'
    template_file = None
    abc = None
    prim_vec = 1
    sweep = m.LatticeParameterSweep(energy_driver, template_file, abc, prim_vec=prim_vec)
    os.chdir(test_dir) 
    shutil.copy(os.path.join(input_dir, 'log.structurerelax.example'), 'log')
    E = sweep.get_energy()
    os.remove('log')
    os.chdir('..')
    assert np.isclose(E, -7.3655263854)
    
def test_fit_to_murnaghan():
    assert False


def test_run_energy_calculations_abinit2():
    """
    sets up files and runs abinit for each lattice constant
    
    Requires abinit set up correctly
    """
    with TemporaryDirectory() as tmp_dir:
        # set up example input files in temporary directory
        os.chdir(tmp_dir)
        os.mkdir('templatedir')
        shutil.copy(os.path.join(input_dir, 'files.example.Li'), 
                    os.path.join('templatedir', 'files'))

        # run sweep  in tmp_dir
        energy_driver = 'abinit'
        template_file = os.path.join(input_dir, 'abinit.in.template.Li')
        abc = np.array([[a*pert for a in 3*[3.3]] for pert in [0.95, 1, 1.05]])
        prim_vec = [[0.5,0.5,-0.5], [-0.5,0.5,0.5], [0.5,-0.5,0.5]]
        sweep = m.LatticeParameterSweep(energy_driver, template_file, abc, prim_vec=prim_vec)
        sweep.run_energy_calculations()
    pass # just gonna assume if it gets to here without error, then nothing is wrong
