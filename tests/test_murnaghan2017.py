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
        print self.name
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.start_dir)
        #shutil.rmtree(self.name)

def test_angles_prim_vec_neither():
    """
    raises value error if neither of angles and prim_vec are specified
    """
    energy_driver = None
    template_file = None
    #abc_guess = []
    #s = []
    abc = []
    with pytest.raises(ValueError):
        m.DftRun(energy_driver, template_file, abc)

def test_angles_prim_vec_both():
    """
    raises value error if both angles and prim_vec are specified
    """
    energy_driver = None
    template_file = None
    abc = []
    with pytest.raises(ValueError):
        m.DftRun(energy_driver, template_file, abc, angles=[90, 90, 90], prim_vec_unscaled=[] )

def test_preprocess_file_bad():
    """
    entering bad name for energy driver raises value error
    """
    energy_driver = 'bad_energy_driver_name'
    template_file = None
    pvu = []
    abc = []
    run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
    with pytest.raises(ValueError):
        run._preprocess_file()

def test_preprocess_file_abinit_angdeg():
    """
    calling _preprocess_file with for abinit correctly writes lattice constants and angles to abinit input file.
    
    not very robust since different number format would cause a fail
    """
    with TemporaryDirectory() as tmp_dir:
        energy_driver = 'abinit'
        template_file = os.path.join(input_dir, 'abinit.in.template.example')
        correct_file = os.path.join(input_dir, 'abinit.in.correct')
        ang = [90, 90, 120]
        abc = [1.5, 3.0, 4.5]
        run = m.DftRun(energy_driver, template_file, abc, angles=ang)
        run._preprocess_file()
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
        abc = [1.5,3,6]
        pvu = [[1,1,0], [0,1,1], [1,0,1]]
        run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
        run._preprocess_file()
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
        abc = [1.5, 3, 6] 
        pvu = [[1,1,0], [0,1,1], [1,0,1]]
        run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
        run._preprocess_file()
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
        abc = [1.5, 3, 6] 
        pvu = [[1,1,0], [0,1,1], [1,0,1]]
        run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
        run._preprocess_file()
        # compare written elk.in file with correct input file
        with open(correct_file) as f1, open('elk.in') as f2:
            assert f1.readlines() == f2.readlines()

def test_prim_vec_from_angles_hex():
    energy_driver = None
    template_file = None
    #s = []
    abc = []
    angles = [90, 90, 120]
    run = m.DftRun(energy_driver, template_file, abc, angles=angles)
    correct_vec = np.array([[1,0,0],[-0.5, np.sqrt(3)/2, 0], [0,0,1]])
    assert np.isclose(run._prim_vec_from_angles(), correct_vec).all()

def test_prim_vec_from_angles_tetr():
    energy_driver = None
    template_file = None
    #s = []
    abc = []
    angles = [90, 90, 90]
    run = m.DftRun(energy_driver, template_file, abc, angles=angles)
    correct_vec = np.array([[1,0,0],[0,1,0], [0,0,1]])
    assert np.isclose(run._prim_vec_from_angles(), correct_vec).all()

def test_get_energy_bad():
    """
    entering bad name for energy driver raises value error
    """
    energy_driver = 'bad'
    template_file = None
    abc = []
    pvu = [[1,1,0], [0,1,1], [1,0,1]]
    run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
    with pytest.raises(ValueError):
        run.get_energy()

def test_abinit_get_energy_single():
    """When only single etotal reported (typical case), returns etotal in Hartree"""
    with TemporaryDirectory() as tmp_dir:
        energy_driver='abinit'
        template_file = None
        abc = []
        pvu = []
        run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
        shutil.copy(os.path.join(input_dir, 'log.example'), 'log')
        E = run.get_energy()
        assert np.isclose(E, -6.1395717098)
    

def test_abinit_get_energy_multiple():
    """When multiple etotal reported, returns correct etotal in Hartree"""
    with TemporaryDirectory() as tmp_dir:
        energy_driver='abinit'
        template_file = None
        abc = []
        pvu = []
        run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
        shutil.copy(os.path.join(input_dir, 'log.structurerelax.example'), 'log')
        E = run.get_energy()
        assert np.isclose(E, -7.3655263854)

def test_get_energy_socorro():
    """read cell enrgy from socorro correctly"""
    with TemporaryDirectory() as tmp_dir:
        energy_driver = 'socorro'
        template_file = None
        abc = []
        pvu = []
        run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
        shutil.copy(os.path.join(input_dir, 'diaryf.example'), 'diaryf')
        E = run.get_energy()
        assert np.isclose(E, -74.637868487/2.)

def test_get_energy_socorro_none():
    """ raises NoEnergyFromDFT if socorro doesn't complete """
    with TemporaryDirectory() as tmp_dir:
        with pytest.raises(m.NoEnergyFromDFT):
            energy_driver = 'socorro'
            template_file = None
            abc = []
            pvu = []
            run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
            shutil.copy(os.path.join(input_dir, 'diaryf.noenergy'), 'diaryf')
            E = run.get_energy()

def test_get_energy_socorro_relax():
    """ gets FINAL energy from socorro diaryf if relaxation was performed """
    with TemporaryDirectory() as tmp_dir:
        energy_driver = 'socorro'
        template_file = None
        abc = []
        pvu = []
        run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
        shutil.copy(os.path.join(input_dir, 'diaryf.relax'), 'diaryf')
        E = run.get_energy()
        assert np.isclose(E, -360.361967002/2.)
    
    
def test_get_energy_elk():
    """read cell enrgy from socorro correctly"""
    with TemporaryDirectory() as tmp_dir:
        energy_driver = 'elk'
        template_file = None
        abc = []
        pvu = []
        run = m.DftRun(energy_driver, template_file, abc, prim_vec_unscaled=pvu)
        shutil.copy(os.path.join(input_dir, 'TOTENERGY.OUT.example'), 'TOTENERGY.OUT')
        E = run.get_energy()
        assert np.isclose(E, -7.51826352965)
    

def test_calc_unit_cell_volume():
    abc = [1,2,3]
    pvu = [[1,0,0], [0,1,0], [0,0,1]]
    run = m.DftRun(None, None, abc, prim_vec_unscaled=pvu)
    assert np.isclose(run._calc_unit_cell_volume(),  6.)
    
def test_calc_unit_cell_volume_2():
    abc = [8,10,12]
    pvu = [[2,0,0], [0,3,0], [0,0,1]]
    run = m.DftRun(None, None, abc, prim_vec_unscaled=pvu)
    assert np.isclose(run._calc_unit_cell_volume(),  5760.)
    
def test_calc_unit_cell_volume_3():
    abc = [12,15,18]
    pvu = [[1,1,0], [0,1,1], [1,0,1]]
    run = m.DftRun(None, None, abc, prim_vec_unscaled=pvu)
    assert np.isclose(run._calc_unit_cell_volume(),  6480.)
    
def test_calc_unit_cell_volume_5():
    abc = [3,6,9]
    angles = [90,90,90]
    run = m.DftRun(None, None, abc, angles=angles)
    assert np.isclose(run._calc_unit_cell_volume(),  162.)
    
def test_calc_unit_cell_volume_6():
    abc = [2,2,10]
    angles = [90,90,120]
    run = m.DftRun(None, None, abc, angles=angles)
    assert np.isclose(run._calc_unit_cell_volume(), 20*np.sqrt(3))
    
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
        correct_file = os.path.join(input_dir, 'energies.dat.correct')
        s = [0.95, 1, 1.1]
        abc_guess = [1,1,2]
        abc_list = [si*np.array(abc_guess) for si in s]
        pvu = [[1,1,0], [0,1,1], [np.sqrt(3),0,np.sqrt(3)/2]]
        volumes = [1.234567890, 2.345678901, 3.456789012] # fake data
        energy_list_hartree = [4.567890123, 5.678901234, 6.789012345] # fake data
        m.write_energy_data(pvu, abc_list, volumes, energy_list_hartree)
        with open(correct_file) as f1, open('energies.dat') as f2:
            assert f1.readlines() == f2.readlines()

def test_abc_of_vol():
    abc_guess = [1,2,3]
    assert np.isclose(m.abc_of_vol(6, 6, abc_guess), [1,2,3]).all()
    
def test_abc_of_vol2():
    abc_guess = [1,2,3]
    assert np.isclose(m.abc_of_vol(48, 6, abc_guess), [2,4,6]).all()

def test_Poly2Dfit():
    x = [5.850600000, 5.850600000, 5.850600000, 5.850600000, 5.850600000,
         5.910300000, 5.910300000, 5.910300000, 5.910300000, 5.910300000,
         5.970000000, 5.970000000, 5.970000000, 5.970000000, 5.970000000,
         6.029700000, 6.029700000, 6.029700000, 6.029700000, 6.029700000,
         6.089400000, 6.089400000, 6.089400000, 6.089400000, 6.089400000]
    y = [9.506000000, 9.603000000, 9.700000000, 9.797000000, 9.894000000,
         9.506000000, 9.603000000, 9.700000000, 9.797000000, 9.894000000,
         9.506000000, 9.603000000, 9.700000000, 9.797000000, 9.894000000,
         9.506000000, 9.603000000, 9.700000000, 9.797000000, 9.894000000,
         9.506000000, 9.603000000, 9.700000000, 9.797000000, 9.894000000]
    z = [-637.087902252, -637.089311103, -637.090170987, -637.090445458, -637.090226054,
         -637.090573150, -637.091802939, -637.092481531, -637.092640128, -637.092279140,
         -637.092024592, -637.093100621, -637.093620462, -637.093592100, -637.093099222,
         -637.092325416, -637.093229113, -637.093620853, -637.093480400, -637.092878369,
         -637.091598554, -637.092424577, -637.092614509, -637.092331806, -637.091584973]
    polynomial_coefficients_correct = [-5.96833484e+02, -1.21294313e+01, -3.97365056e+00,  1.62409213e+00,
                                        2.82249323e-01,  2.86799395e-01, -7.70793542e-02, -9.01258094e-03,
                                       -7.69390584e-03, -7.31762636e-03]
    equilibrium_lattice_parameters_correct =  [5.99737564, 9.7344405]
    E_min_correct = -637.093782357
    fit = m.Poly2DFit(x, y, z)
    assert np.isclose(fit.coeff, polynomial_coefficients_correct).all()
    assert np.isclose(fit.min_xy, equilibrium_lattice_parameters_correct).all()
    assert np.isclose(fit.min_E, -637.093810214)


def test_write_murnaghan_data():
    with TemporaryDirectory() as tmp_dir:
        correct_file = os.path.join(input_dir, 'murnaghan_parameters.dat.correct')
        s = [0.95, 0.975, 1, 1.025, 1.05]
        abc_guess = [6.6, 6.6, 6.6]
        abc_list = [si*np.array(abc_guess) for si in s]
        pvu = [[0.5,0.5,-0.5], [-0.5,0.5,0.5], [0.5,-0.5,0.5]]
        volumes = [94.091762000, 101.717255250, 109.744000000, 118.182284750, 127.042398000]
        energies_hartree = [-7.515024550, -7.516721665, -7.517852086, -7.518522990, -7.518821239]
        fit = m.MurnaghanFit(volumes, energies_hartree)
        fit.E0 = -7.518851002  # fake results
        fit.B0 = 0.000433261789  # fake results
        fit.BP = 3.505747298  # fake results
        fit.V0 = 131.192891163  # fake results
        m.write_murnaghan_data(fit, volumes, abc_list)
        with open(correct_file) as f1, open('murnaghan_parameters.dat') as f2:
            assert f1.readlines() == f2.readlines()


def test_write_Poly2D_data():
    with TemporaryDirectory() as tmp_dir:
        correct_file = os.path.join(input_dir, 'poly2d_parameters.dat.correct')
        fit = type('', (), {})()  # fake object
        fit.coeff = [-4.65217793e+02, -5.57206712e+01, -3.38089647e+01,
                     4.58462590e+00, -9.04248330e-01,  4.61461726e-02,
                     1.71821865e+00, -5.58679207e-01,  1.09713642e+01, 1.234]
        fit.min_xy =  [5.9997302, 9.73756115]
        fit.min_E = -637.093810214
        m.write_poly2d_data(fit)
        with open(correct_file) as f1, open('poly2d_parameters.dat') as f2:
            assert f1.readlines() == f2.readlines()


def test_integration_elk():
    """
    lattice paramter sweep and murnaghan fitting should run correctly
    
    Requires elk set up correctly. Also, this test is fragile because
    different elk versions could calulate different energies. If this causes
    problems in the future, either increase np.isclose tolerance or (worse) update
    energy values to new elk outputs.
    If the energy values are wrong, the murnaghan paramters will be too.
    """
    with TemporaryDirectory() as tmp_dir:
        # set up example input files in temporary directory
        os.mkdir('templatedir')
        shutil.copy(os.path.join(input_dir, 'elk.in.template.Li'), 
                    os.path.join('templatedir', 'elk.in.template'))
        shutil.copy(os.path.join(input_dir, 'Li.in.elkspecies'), 
                    os.path.join('templatedir', 'Li.in'))
        # run sweep  in tmp_dir
        energy_driver = 'elk'
        template_file = 'elk.in.template'
        s = [0.95, 0.975, 1, 1.025, 1.05]
        abc_guess = [7.6, 7.6, 7.6]
        abc_list = [si*np.array(abc_guess) for si in s]
        pvu = [[0.5,0.5,0.0], [0.0,0.5,0.5], [0.5,0.0,0.5]]
        volumes, energy_list_hartree = m.lattice_parameter_sweep(energy_driver, template_file, abc_list, prim_vec_unscaled=pvu)

        fit = m.MurnaghanFit(volumes, energy_list_hartree)
        m.write_murnaghan_data(fit, volumes, abc_list)  # don't really need to do this
        # assert data files written (correctly or not)
        assert os.path.exists('energies.dat')
        assert os.path.exists('murnaghan_parameters.dat')

    # assert volumes and energies are correct
    assert np.isclose(volumes, [94.091762000, 101.717255250, 109.744000000, 118.182284750, 127.042398000]).all()
    assert np.isclose(energy_list_hartree, [-7.515024699, -7.516721695, -7.517852094, -7.518522940, -7.518821210], atol=1e-3, rtol=0).all()
    # assert murnaghan parameters are correct
    assert np.isclose(fit.E0, -7.518850974, atol=1e-4, rtol=0)
    assert np.isclose(fit.B0, 0.00043323874, atol=1e-5, rtol=0)
    assert np.isclose(fit.BP, 3.505576932, atol=1e-2, rtol=0)
    assert np.isclose(fit.V0, 131.193402033, atol=1e0, rtol=0)


def test_integration_socorro():
    """
    lattice paramter sweep and murnaghan fitting should run correctly
    
    Requires socorro set up correctly. Also, this test is fragile because
    different socorro versions could calulate different energies. If this causes
    problems in the future, either increase np.isclose tolerance or (worse) update
    energy values to new socorro outputs.
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
        abc_list = [si*np.array(abc_guess) for si in s]
        pvu = [[0.5,0.5,0.0], [0.0,0.5,0.5], [0.5,0.0,0.5]]
        volumes, energy_list_hartree = m.lattice_parameter_sweep(energy_driver, template_file, abc_list, prim_vec_unscaled=pvu)

        fit = m.MurnaghanFit(volumes, energy_list_hartree)
        m.write_murnaghan_data(fit, volumes, abc_list)  # don't really need to do this
        # assert data files written (correctly or not)
        assert os.path.exists('energies.dat')
        assert os.path.exists('murnaghan_parameters.dat')

    # assert volumes and energies are correct
    assert np.isclose(volumes, [94.091762000, 101.717255250, 109.744000000, 118.182284750, 127.042398000]).all()
    assert np.isclose(energy_list_hartree, [-37.012225020, -37.042712697, -37.067941736, -37.090042723, -37.106637075], atol=1e-5, rtol=0).all()
    # assert murnaghan parameters are correct
    assert np.isclose(fit.E0, -37.129044175, atol=1e-4, rtol=0)
    assert np.isclose(fit.B0, 0.00722071666, atol=1e-5, rtol=0)
    assert np.isclose(fit.BP, 0.643620090, atol=1e-2, rtol=0)
    assert np.isclose(fit.V0, 156.473079733, atol=1e-1, rtol=0)


def test_integration_socorro_Poly2D():
    """
    lattice paramter sweep and 2D polynomial fitting should run correctly
    
    Requires socorro set up correctly. Also, this test is fragile because
    different socorro versions could calulate different energies. If this causes
    problems in the future, either increase np.isclose tolerance or (worse) update
    energy values to new socorro outputs.
    If the energy values are wrong, the fit paramters will be too.
    """
    with TemporaryDirectory() as tmp_dir:
        # set up example input files in temporary directory
        shutil.copytree(os.path.join(input_dir, 'socorro_GaN'), 'templatedir')
        energy_driver = 'socorro'
        template_file = 'crystal.template'
        # create list of a,b,c values to test
        s_a = [0.995, 1.0, 1.005, 1.01]  # scales for a/b lattice parameters
        s_c = [0.995, 1.0, 1.005, 1.01]  # scales for c lattice parameter
        abc_guess = np.array([5.97, 5.97, 9.7])
        abc_list = []
        for s1 in s_a:
            for s2 in s_c:
                abc_list.append(np.array([s1, s1, s2]) * abc_guess)
        abc_list = np.array(abc_list)

        angle = 120.*np.pi/180.
        pvu = [[1.0,  0.0,  0.0],
               [np.cos(angle),  np.sin(angle),  0.0],
               [0.0,  0.0,  1.0]]
        volumes, energy_list_hartree = m.lattice_parameter_sweep(energy_driver, template_file, abc_list, prim_vec_unscaled=pvu)

        a_values = abc_list[:,0]
        c_values = abc_list[:,2]
        fit = m.Poly2DFit(a_values, c_values, energy_list_hartree)
        m.write_poly2d_data(fit)

        # assert data files written (correctly or not)
        assert os.path.exists('energies.dat')
        assert os.path.exists('poly2d_parameters.dat')

    # assert volumes and energies are correct
    correct_volumes = [294.930896168, 296.412960973, 297.895025778, 299.377090583,
                       297.902473340, 299.399470693, 300.896468047, 302.393465400,
                       300.888945635, 302.400950387, 303.912955139, 305.424959891,
                       303.890313054, 305.417400054, 306.944487054, 308.471574055]
    correct_energies = [-637.080005241, -637.080245315, -637.080411884, -637.080209983,
                        -637.080415094, -637.080655151, -637.080667730, -637.080582546,
                        -637.080643365, -637.080848369, -637.080823277, -637.080714608,
                        -637.080655319, -637.080861547, -637.080842684, -637.080622386]
    assert np.isclose(volumes, correct_volumes).all()
    assert np.isclose(energy_list_hartree, correct_energies, atol=1e-5, rtol=0).all()

    # assert 2D poly fit parameters are correct
    print fit.coeff
    print fit.min_xy
    print fit.min_E
    assert np.isclose(fit.coeff, [-629.071534722, 2.300932235, -2.977501186, 
        -0.622605148, 0.141914732, 0.215972091, 0.027849296, 0.023394877, 
        -0.020622472, -0.001799476], atol=1e-3, rtol=0).all()
    assert np.isclose(fit.min_xy, [6.012510409, 9.725896217], atol=1e-5, rtol=0).all()
    assert np.isclose(fit.min_E, -637.080891910, atol=1e-5, rtol=0)


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
        abc_list = [si*np.array(abc_guess) for si in s]
        pvu = [[0.5,0.5,-0.5], [-0.5,0.5,0.5], [0.5,-0.5,0.5]]
        volumes, energy_list_hartree = m.lattice_parameter_sweep(energy_driver, template_file, abc_list, prim_vec_unscaled=pvu)

        fit = m.MurnaghanFit(volumes, energy_list_hartree)
        m.write_murnaghan_data(fit, volumes, abc_list)  # don't really need to do this
        # assert data files written (correctly or not)
        # assert data files written (correctly or not)
        assert os.path.exists('energies.dat')
        assert os.path.exists('murnaghan_parameters.dat')

    # assert volumes and energies are correct
    assert np.isclose(volumes, [15.40574269, 16.65427268, 17.9685, 19.3501092, 20.80078481]).all()
    assert np.isclose(energy_list_hartree, [-5.93211143, -6.04145629, -6.13957171, -6.22787361, -6.30747694], atol=1e-3, rtol=0).all()
    # assert murnaghan parameters are correct
    assert np.isclose(fit.E0, -6.7617490608, atol=1e-4, rtol=0)
    assert np.isclose(fit.B0, 0.02539667138, atol=1e-4, rtol=0)
    assert np.isclose(fit.BP, 1.71091944591, atol=1e-1, rtol=0)
    assert np.isclose(fit.V0, 49.4869248327, atol=1e-2, rtol=0)
