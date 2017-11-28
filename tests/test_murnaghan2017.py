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
        self.name = tempfile.mkdtemp(dir=main_dir)
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
    abc_guess = []
    s = []
    with pytest.raises(ValueError):
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess)

def test_angles_prim_vec_both():
    """
    raises value error if both angles and prim_vec are specified
    """
    energy_driver = None
    template_file = None
    abc_guess = []
    s = []
    with pytest.raises(ValueError):
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess, angles=1, prim_vec_unscaled=[])

def test_angles_None_when_prim_vec_specified():
    """
    if prim_vec is specified, self.angles=None
    """
    energy_driver = None
    template_file = None
    s = []
    abc_guess = []
    sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess, prim_vec_unscaled=[])
    assert sweep.angles is None

def test_preprocess_file_bad():
    """
    entering bad name for energy driver raises value error
    """
    energy_driver = 'dummy'
    template_file = None
    s = [0.5, 1.0, 1.5]
    abc_guess = [1,2,3]
    pvu = [[1,1,0], [0,1,1], [1,0,1]]
    sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess, prim_vec_unscaled=pvu)
    with pytest.raises(ValueError):
        sweep._preprocess_file(0)

def test_preprocess_file_abinit_angdeg():
    """
    calling _preprocess_file with for abinit correctly writes lattice constants and angles to abinit input file.
    
    not very robust since different number format would cause a fail
    """
    with TemporaryDirectory() as tmp_dir:
        energy_driver = 'abinit'
        template_file = os.path.join(input_dir, 'abinit.in.template.example')
        correct_file = os.path.join(input_dir, 'abinit.in.correct')
        s = [0.9, 1.0, 1.5]
        abc_guess = [1,2,3]
        ang = [90, 90, 120]
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess, angles=ang)
        sweep._preprocess_file(2)
        # compare written abinit file with correct input file
        with open(correct_file) as f1, open('abinit.in') as f2:
            assert f1.readlines() == f2.readlines()


def test_preprocess_file_abinit_rprim():
    """
    calling _preprocess_file with for abinit correctly writes acell and lattice vectors to file
    
    not very robust since different number format would cause a fail
    """
    with TemporaryDirectory() as tmp_dir:
        correct_file = os.path.join(input_dir, 'abinit.in.correct.2')
        energy_driver = 'abinit'
        template_file = os.path.join(input_dir, 'abinit.in.template.example')
        s = [0.9, 1.0, 1.5]
        abc_guess = [1,2,4]
        pvu = [[1,1,0], [0,1,1], [1,0,1]]
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess=abc_guess, prim_vec_unscaled=pvu)
        sweep._preprocess_file(2)
        # compare written abinit file with correct input file
        with open(correct_file) as f1, open('abinit.in') as f2:
            assert f1.readlines() == f2.readlines()


def test_preprocess_file_socorro_rprim():
    """
    calling _preprocess_file for socorro correctly writes scale and lattice constants to socorro crystal file
    
    not very robust since different number format would cause a fail
    """
    with TemporaryDirectory() as tmp_dir:
        correct_file = os.path.join(input_dir, 'crystal.correct')
        energy_driver = 'socorro'
        template_file = os.path.join(input_dir, 'crystal.template.example')
        s = [0.9, 1.0, 1.5]
        abc_guess = [1,2,4]
        pvu = [[1,1,0], [0,1,1], [1,0,1]]
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess=abc_guess, prim_vec_unscaled=pvu)
        sweep._preprocess_file(2)
        # compare written socorro crystal file with correct input file
        with open(correct_file) as f1, open('crystal') as f2:
            assert f1.readlines() == f2.readlines()


def test_preprocess_file_elk_rprim():
    """
    calling _preprocess_file for elk correctly writes scale and lattice constants to elk.in
    
    not very robust since different number format would cause a fail
    """
    with TemporaryDirectory() as tmp_dir:
        correct_file = os.path.join(input_dir, 'elk.in.correct')
        energy_driver = 'elk'
        template_file = os.path.join(input_dir, 'elk.in.template.example')
        s = [0.9, 1.0, 1.5]
        abc_guess = [1,2,4]
        pvu = [[1,1,0], [0,1,1], [1,0,1]]
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess=abc_guess, prim_vec_unscaled=pvu)
        sweep._preprocess_file(2)
        # compare written elk.in file with correct input file
        with open(correct_file) as f1, open('elk.in') as f2:
            assert f1.readlines() == f2.readlines()



def test_prim_vec_from_angles_hex():
    energy_driver = None
    template_file = None
    s = []
    abc = []
    angles = [90, 90, 120]
    sweep = m.LatticeParameterSweep(energy_driver, template_file, abc, s, angles=angles)
    correct_vec = np.array([[1,0,0],[-0.5, np.sqrt(3)/2, 0], [0,0,1]])
    assert np.isclose(sweep._prim_vec_from_angles(), correct_vec).all()

def test_prim_vec_from_angles_tetr():
    energy_driver = None
    template_file = None
    s = []
    abc = []
    angles = [90, 90, 90]
    sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc, angles=angles)
    correct_vec = np.array([[1,0,0],[0,1,0], [0,0,1]])
    assert np.isclose(sweep._prim_vec_from_angles(), correct_vec).all()

def test_get_energy_bad():
    """
    entering bad name for energy driver raises value error
    """
    energy_driver = 'bad'
    template_file = None
    s = []
    abc = []
    pvu = [[1,1,0], [0,1,1], [1,0,1]]
    sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc, prim_vec_unscaled=pvu)
    with pytest.raises(ValueError):
        sweep.get_energy()

def test_abinit_get_energy_single():
    """When only single etotal reported (typical case), returns etotal in Hartree"""
    with TemporaryDirectory() as tmp_dir:
        energy_driver='abinit'
        template_file = None
        s = []
        abc = []
        pvu = []
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc, prim_vec_unscaled=pvu)
        shutil.copy(os.path.join(input_dir, 'log.example'), 'log')
        E = sweep.get_energy()
        assert np.isclose(E, -6.1395717098)
    

def test_abinit_get_energy_multiple():
    """When only single etotal reported (typical case), returns etotal in Hartree"""
    with TemporaryDirectory() as tmp_dir:
        energy_driver='abinit'
        template_file = None
        s = []
        abc = []
        pvu = []
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc, prim_vec_unscaled=pvu)
        shutil.copy(os.path.join(input_dir, 'log.structurerelax.example'), 'log')
        E = sweep.get_energy()
        assert np.isclose(E, -7.3655263854)

def test_get_energy_socorro():
    """read cell enrgy from socorro correctly"""
    with TemporaryDirectory() as tmp_dir:
        energy_driver = 'socorro'
        template_file = None
        s = []
        abc = []
        pvu = []
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc, prim_vec_unscaled=pvu)
        shutil.copy(os.path.join(input_dir, 'diaryf.example'), 'diaryf')
        E = sweep.get_energy()
        assert np.isclose(E, -74.637868487/2.)

def test_get_energy_elk():
    """read cell enrgy from socorro correctly"""
    with TemporaryDirectory() as tmp_dir:
        energy_driver = 'elk'
        template_file = None
        s = []
        abc = []
        pvu = []
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc, prim_vec_unscaled=pvu)
        shutil.copy(os.path.join(input_dir, 'TOTENERGY.OUT.example'), 'TOTENERGY.OUT')
        E = sweep.get_energy()
        assert np.isclose(E, -7.51826352965)
    

def test_calc_unit_cell_volume():
    s = [1]
    abc_guess = [1,2,3]
    pvu = [[1,0,0], [0,1,0], [0,0,1]]
    sweep = m.LatticeParameterSweep(None, None, s, abc_guess, prim_vec_unscaled=pvu)
    assert np.isclose(sweep._calc_unit_cell_volume(0),  6.)
    
def test_calc_unit_cell_volume_2():
    s = [1,2]
    abc_guess = [4,5,6]
    pvu = [[2,0,0], [0,3,0], [0,0,1]]
    sweep = m.LatticeParameterSweep(None, None, s, abc_guess, prim_vec_unscaled=pvu)
    assert np.isclose(sweep._calc_unit_cell_volume(1),  5760.)
    
def test_calc_unit_cell_volume_3():
    s = [3]
    abc_guess = [4,5,6]
    pvu = [[1,1,0], [0,1,1], [1,0,1]]
    sweep = m.LatticeParameterSweep(None, None, s, abc_guess, prim_vec_unscaled=pvu)
    assert np.isclose(sweep._calc_unit_cell_volume(0),  6480.)
    
def test_calc_unit_cell_volume_5():
    s = [3]
    abc_guess = [1,2,3]
    angles = [90,90,90]
    sweep = m.LatticeParameterSweep(None, None, s, abc_guess, angles=angles)
    assert np.isclose(sweep._calc_unit_cell_volume(0),  162.)
    
def test_calc_unit_cell_volume_6():
    s = [1]
    abc_guess = [2,2,10]
    angles = [90,90,120]
    sweep = m.LatticeParameterSweep(None, None, s, abc_guess, angles=angles)
    assert np.isclose(sweep._calc_unit_cell_volume(0), 20*np.sqrt(3))
    
def test_murnaghan_equation():
    parameters = [1, 2, 3, 4]
    vol_list = [0.1, 1, 10]
    assert np.isclose(m.murnaghan_equation(parameters, vol_list), [2130.4, 19, 3.88]).all()

def test_MurnaghanFit():
    vol_array = [34.598173802, 37.402118963, 40.353607000, 43.456421063, 46.714344303]
    E_array = [-2.126149418, -2.130179030, -2.132207477, -2.132510305, -2.131351938]
    testfit = m.MurnaghanFit(vol_array, E_array)
    assert np.isclose(testfit.E0, -2.132584858)
    assert np.isclose(testfit.B0, 0.00665127134)
    assert np.isclose(testfit.BP, 2.997968514)
    assert np.isclose(testfit.V0, 42.487211639)

def test_write_energy_data():
    with TemporaryDirectory() as tmp_dir:
        s = [0.95, 1, 1.1]
        abc_guess = [1,1,2]
        pvu = [[1,1,0], [0,1,1], [np.sqrt(3),0,np.sqrt(3)/2]]
        sweep = m.LatticeParameterSweep('', '', s, abc_guess, prim_vec_unscaled=pvu)
        sweep.volumes = [1.234567890, 2.345678901, 3.456789012] # fake data
        sweep.energies_hartree = [4.567890123, 5.678901234, 6.789012345] # fake data
        sweep._write_energy_data()
    pass

def test_abc_of_vol():
    abc_guess = [1,2,3]
    assert np.isclose(m.abc_of_vol(6, 6, abc_guess), [1,2,3]).all()
    
def test_abc_of_vol2():
    abc_guess = [1,2,3]
    assert np.isclose(m.abc_of_vol(48, 6, abc_guess), [2,4,6]).all()

def test_write_murnaghan_data():
    with TemporaryDirectory() as tmp_dir:
        s = [0.95, 0.975, 1, 1.025, 1.05]
        abc_guess = [6.6, 6.6, 6.6]
        pvu = [[0.5,0.5,-0.5], [-0.5,0.5,0.5], [0.5,-0.5,0.5]]
        sweep = m.LatticeParameterSweep('', '', s, abc_guess, prim_vec_unscaled=pvu)
        sweep.volumes = [15.405742687, 16.654272680, 17.968500000, 19.350109195, 20.800784812] # fake data
        sweep.energies_hartree = [-5.932111434, -6.041456293, -6.139571710, -6.227873609, -6.307476940] # fake data
        sweep.murnaghan_fit = sweep._fit_sweep_to_murnaghan()
        sweep._write_murnaghan_data()
    pass

def test_integration_abinit():
    """
    lattice paramter sweep and murnaghan fitting should run correctly
    
    Requires abinit set up correctly. Also, this test is fragile because
    different abinit versions could calulate different energies. If this causes
    problems in the future, either increase np.isclose tolerance or (worse) update
    energy values to new abinit outputs.
    If the energy values are wrong, the murnaghan paramters will be too.
    """
    with TemporaryDirectory() as tmp_dir:
        # set up example input files in temporary directory
        os.mkdir('templatedir')
        shutil.copy(os.path.join(input_dir, 'files.example.Li'), 
                    os.path.join('templatedir', 'files'))
        shutil.copy(os.path.join(input_dir, 'Li.PAW.abinit'),
                    os.path.join('templatedir', 'Li.PAW.abinit'))

        # run sweep  in tmp_dir
        energy_driver = 'abinit'
        template_file = os.path.join(input_dir, 'abinit.in.template.Li')
        s = [0.95, 0.975, 1, 1.025, 1.05]
        abc_guess = [3.3, 3.3, 3.3]
        pvu = [[0.5,0.5,-0.5], [-0.5,0.5,0.5], [0.5,-0.5,0.5]]
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess, prim_vec_unscaled=pvu)
        sweep.run_energy_calculations()
        #shutil.copy('energies.dat', '..')
        #shutil.copy('murnaghan_parameters.dat', '..')
        # assert data files written (correctly or not)
        assert os.path.exists('energies.dat')
        assert os.path.exists('murnaghan_parameters.dat')

    # assert volumes and energies are correct
    assert np.isclose(sweep.volumes, [15.40574269, 16.65427268, 17.9685, 19.3501092, 20.80078481]).all()
    assert np.isclose(sweep.energies_hartree, [-5.93211143, -6.04145629, -6.13957171, -6.22787361, -6.30747694]).all()
    # assert murnaghan parameters are correct
    assert np.isclose(sweep.murnaghan_fit.E0, -6.76174906082)
    assert np.isclose(sweep.murnaghan_fit.B0, 0.0253966713811)
    assert np.isclose(sweep.murnaghan_fit.BP, 1.71091944591)
    assert np.isclose(sweep.murnaghan_fit.V0, 49.4869248327)


def test_integration_socorro():
    """
    lattice paramter sweep and murnaghan fitting should run correctly
    
    Requires socorro set up correctly. Also, this test is fragile because
    different socorro versions could calulate different energies. If this causes
    problems in the future, either increase np.isclose tolerance or (worse) update
    energy values to new abinit outputs.
    If the energy values are wrong, the murnaghan paramters will be too.
    """
    with TemporaryDirectory() as tmp_dir:
        # set up example input files in temporary directory
        shutil.copytree(os.path.join(input_dir, 'socorro_LiF'), 'templatedir')
        # run sweep  in tmp_dir
        energy_driver = 'socorro'
        template_file = 'crystal.template'
        s = [0.95, 0.975, 1, 1.025, 1.05]
        abc_guess = [7.6, 7.6, 7.6]
        pvu = [[0.5,0.5,0.0], [0.0,0.5,0.5], [0.5,0.0,0.5]]
        sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess, prim_vec_unscaled=pvu)
        sweep.run_energy_calculations()
        #shutil.copy('energies.dat', '..')
        #shutil.copy('murnaghan_parameters.dat', '..')
        # assert data files written (correctly or not)
        assert os.path.exists('energies.dat')
        assert os.path.exists('murnaghan_parameters.dat')

    # assert volumes and energies are correct
    assert np.isclose(sweep.volumes, [94.091762000, 101.717255250, 109.744000000, 118.182284750, 127.042398000]).all()
    assert np.isclose(sweep.energies_hartree, [-37.012225020, -37.042712697, -37.067941736, -37.090042723, -37.106637075]).all()
    # assert murnaghan parameters are correct
    assert np.isclose(sweep.murnaghan_fit.E0, -37.129044175)
    assert np.isclose(sweep.murnaghan_fit.B0, 0.00722071666)
    assert np.isclose(sweep.murnaghan_fit.BP, 0.643620090)
    assert np.isclose(sweep.murnaghan_fit.V0, 156.473079733)

def test_integration_elk():
    assert False
