#!/bin/env python
import shutil
import os 
import sys
import glob
import scipy.optimize
import subprocess
import numpy as np


class NoEnergyFromDFT(Exception):
    """raised if total energy not found in DFT output"""

def lattice_parameter_sweep(energy_driver, template_file, abc_list, angles=None, 
    prim_vec_unscaled=None, dft_command=None):
    """ Runs DFT code at specified list of lattice parameters and returns
    energies and volumes

    Args:
        energy_driver (str): DFT code to run: 'elk', 'socorro', 'abinit'
        template_file (str): template file name. Usually 
            'abinit.in.template' for abinit, 'crystal.template' for socorro, 
            or 'elk.in.template' for socorro.
        abc_list (numpy array of float): N by 3 numpy array containing the 
            lattice vector scales for each set of lattice vectors to test. Each
            row contains a value of a, b, and c lattice parameters.
        angles (3-element array-like, optional): contains the angles between 
            lattice vectors, alpha, beta, and gamma. Either this or 
            prim_vec_unscaled must be specified, but not both.
        prim_vec_unscaled (array-like of float, optional): 3 by 3 array (or 
            list of lists) of the unscaled lattice vectors. Each will be 
            multiplied by the lattice vector scales in abc_list. Either this 
            or angles must be specified, but not both.
        dft_command (str, optional): Command to call dft code, which can include
            flags. This is necessary for running in different cluster 
            environments. Defaults to 'elk', 'abinit', or 'socorro'.

    Returns:
        list of floats: unit cell volume (Bohr^3) at each set of lattice 
            parameters tested.
        list of floats: energy (Ha) at each set of lattice parameters tested.

    Todo:
        Make templatefile a default to the most likely filename for each energy
            driver?
        Pass dft_command from command line. This would make the command more 
            clear and editable in the batch submission script.
    """    
    # remove old work directories
    for f in glob.glob('workdir.*'):
        shutil.rmtree(f)

    # run dft calculations and return calculated energies
    main_dir = os.getcwd()
    energy_list_hartree = []
    volumes = []
    for i in range(len(abc_list)):
        dir_name = 'workdir.'+str(i)
        run = DftRun(energy_driver, template_file, abc_list[i], angles=angles, 
        prim_vec_unscaled=prim_vec_unscaled, dft_command=dft_command)
        run._setup_workdir(dir_name)
        os.chdir(dir_name)
        run._preprocess_file()
        run.run_dft()
        volumes.append(run._calc_unit_cell_volume())
        try:
            energy_list_hartree.append(run.get_energy())
        except NoEnergyFromDFT:
            raise
        finally:
            os.chdir(main_dir)

    write_energy_data(prim_vec_unscaled, abc_list, volumes, energy_list_hartree)
    return volumes, energy_list_hartree


def write_energy_data(prim_vec_unscaled, abc_list, volumes, energy_list_hartree):
    """ 
    Writes raw energy data from dft runs.

    The header should contain primitive vectors. 
    where a,b,c are scaling factors for primitve vectors (bohr)
    V is the volume of the unit cell (Bohr^3)
    E_Ha is the total energy in Hartree
    E_Ry is the total energy in Rydberg
    E_eV is the total energy in eV
    """
    with open('energies.dat', 'w') as fout:
        fout.write('# Note that a, b, and c are scales and need to be multiplied \n'+
                   '# by unscaled primitive vectors prim_vec to get actual unit cell vectors\n')
        fout.write('# unscaled primitive vectors:\n')
        fout.write('# '+str(prim_vec_unscaled[0]) + '\n')
        fout.write('# '+str(prim_vec_unscaled[1]) + '\n')
        fout.write('# '+str(prim_vec_unscaled[2]) + '\n')
        fout.write('# a (bohr),   b (bohr),   c (bohr),     V (bohr^3),    E (Ha),     E (Ry),      E(eV)\n')
        for i in range(len(abc_list)):
            a = abc_list[i][0]
            b = abc_list[i][1]
            c = abc_list[i][2]
            V = volumes[i] 
            E_Ha = energy_list_hartree[i]
            E_Ry = E_Ha*2.
            E_eV = E_Ha*27.21138602
            fout.write('%.9f   %.9f   %.9f   %.9f   %.9f   %.9f   %.9f \n'
                    %(a, b, c, V, E_Ha, E_Ry, E_eV))


class DftRun(object):
    def __init__(self, energy_driver, template_file, abc, angles=None, 
                 prim_vec_unscaled=None, dft_command=None):
        self.energy_driver = energy_driver
        self.template_file = template_file
        self.abc = np.array(map(float, abc))
        self.dft_command = dft_command
        # set prim_vec_unscaled and maybe angles:
        # the unit cell can be specified with either primitive vectors (which are scaled by abc)
        # or angles (useful for hex)
        if prim_vec_unscaled is None and angles is None:
            raise ValueError('Either prim_vec or angles should be defined')
        elif prim_vec_unscaled is not None and angles is not None:
            raise ValueError('Either prim_vec or angles should be defined, but not both')
        elif prim_vec_unscaled is not None and angles is None:
            self.prim_vec_unscaled = np.array([map(float, row) for row in prim_vec_unscaled])
            self.angles = angles # angles don't need to be calculated
        else:
            self.angles = np.array(map(float, angles))
            self.prim_vec_unscaled = self._prim_vec_from_angles()

    def _prim_vec_from_angles(self):
        """
        Given angles between lattice vectors, returns unit vectors

        I chose ahat to be [1,0,0], and bhat to be in the x-y plane.
        """
        alp = self.angles[0]*np.pi/180.
        bet = self.angles[1]*np.pi/180.
        gam = self.angles[2]*np.pi/180.
        b1 = np.cos(gam)
        b2 = np.sin(gam)
        c1 = np.cos(bet)
        c2 = (np.cos(alp) - b1*c1)/b2
        c3 = np.sqrt(1-c1**2-c2**2)
        ahat = [1,0,0]
        bhat = [b1, b2, 0]
        chat = [c1, c2, c3]
        return np.array([ahat, bhat, chat])
    
    def _calc_unit_cell_volume(self):
        """
        Calculate volume of unit cell from primitive vectors.
        (a*b*c) * u.(vxw) where u,v,w are the unscaled prmitive vectors
        and a,b,c are acell[ind]
 
        ind is index of s to use
        """        
        a = self.abc[0]
        b = self.abc[1]
        c = self.abc[2]
        u = self.prim_vec_unscaled[0]
        v = self.prim_vec_unscaled[1]
        w = self.prim_vec_unscaled[2]
        V_unscaled = np.dot(u, np.cross(v,w))
        V = a*b*c*V_unscaled
        return V

    def _setup_workdir(self, dir_name):
        if self.energy_driver=='abinit':
            shutil.copytree('templatedir', dir_name, symlinks=True)
        elif self.energy_driver=='socorro':
            shutil.copytree('templatedir', dir_name, symlinks=True)
            this_dir = os.getcwd()
            os.chdir(dir_name)
            os.mkdir('data')
            os.chdir('data')
            os.symlink(os.path.join('..', 'crystal'), 'crystal')
            for file in [os.path.basename(x) for x in glob.glob('../PAW.*')]:
                os.symlink(os.path.join('..', file), file)
            os.chdir(this_dir)
        elif self.energy_driver=='elk':
            shutil.copytree('templatedir', dir_name, symlinks=True)
        else:
            raise ValueError('Unknown energy driver specified')

    def _preprocess_file(self):
        """
        wraps the specific file preprocess functions for different dft codes
        """
        if self.energy_driver=='abinit':
            self._preprocess_file_abinit()
        elif self.energy_driver=='socorro':
            self._preprocess_file_socorro()
        elif self.energy_driver=='elk':
            self._preprocess_file_elk()
        else:
            raise ValueError('Unknown energy driver specified')

    def _preprocess_file_abinit(self):
        """
        writes abinit.in from template and appends lattice vector lengths (acell) and 
        EITHER angles (angdeg) or primitive vectors (rprim), depending on which was 
        passed in to the LatticeVectorSweep object
        """
        shutil.copy2(self.template_file, 'abinit.in')
        if self.angles is not None:
            with open('abinit.in', 'a') as f:
                f.write('acell ' + ' '.join([str(float(n)) for n in self.abc]) + '\n')
                f.write('angdeg ' + ' '.join([str(float(n)) for n in self.angles]) + '\n')
        else:
            with open('abinit.in', 'a') as f:
                f.write('acell ' + ' '.join([str(float(n)) for n in self.abc]) + '\n')
                f.write('rprim ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[0]]) + '\n')
                f.write('      ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[1]]) + '\n')
                f.write('      ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[2]]) + '\n')

    def _preprocess_file_socorro(self):
        """
        writes crystal file from template with lattice vectors and scale

        Socorro only accepts one scale value (as opposed to 3), so the scale is specified as
        acell[0]
        and the primitve vectors are
        prim_vec_unscaled[0]
        acell[1]/acell[0] * prim_vec_unscaled[1]
        acell[2]/acell[0] * prim_vec_unscaled[2]
        """
        try:
            shutil.copy2(self.template_file, 'crystal.template')
        except shutil.Error:
            pass
        if self.angles is not None:
            with open('crystal', 'a') as f:
                raise ValueError('angdeg input not implemented for socorro yet')
        else:
            scale = self.abc[0]
            prim_vec_0 = self.prim_vec_unscaled[0]
            prim_vec_1 = self.abc[1]/self.abc[0] * self.prim_vec_unscaled[1]
            prim_vec_2 = self.abc[2]/self.abc[0] * self.prim_vec_unscaled[2]
            with open('crystal.template', 'r') as fin:
                template = fin.readlines()
            template.insert(1, '  '+str(scale)+'\n')
            template.insert(2, '    ' + ' '.join(map(str, prim_vec_0)) + '\n')
            template.insert(3, '    ' + ' '.join(map(str, prim_vec_1)) + '\n')
            template.insert(4, '    ' + ' '.join(map(str, prim_vec_2)) + '\n')
            with open('crystal', 'w') as fout:
                for line in template:
                    fout.write(line)

    def _preprocess_file_elk(self):
        """
        writes elk.in from template and appends lattice vector lengths (scale1, scale2, scale3) and 
        primitive vectors (avec)
        """
        shutil.copy2(self.template_file, 'elk.in')
        if self.angles is not None:
            raise ValueError('angdeg input not implemented for elk yet')
        else:
            with open('elk.in', 'a') as f:
                f.write('\nscale1\n  ' + str(float(self.abc[0])) + '\n')
                f.write('\nscale2\n  ' + str(float(self.abc[1])) + '\n')
                f.write('\nscale3\n  ' + str(float(self.abc[2])) + '\n')
                f.write('\navec\n') 
                f.write('  ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[0]]) + '\n')
                f.write('  ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[1]]) + '\n')
                f.write('  ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[2]]) + '\n')

    def run_dft(self):
        """
        runs dft code in current directory

        If dft_run is specified, that command is called. Otherwise, the default
        command for the specified energy_driver is called.
        """
        if self.energy_driver=='abinit':
            with open('log', 'w') as log_fout, open('files','r') as files_fin:
                if self.dft_command is None:
                    subprocess.call(['abinit'], stdin=files_fin, stdout=log_fout)
                else:
                    subprocess.call(self.dft_command.split(), stdin=files_fin, stdout=log_fout)
        elif self.energy_driver=='socorro':
            with open('log', 'w') as log_fout:
                if self.dft_command is None:
                    subprocess.call(['socorro'], stdout=log_fout)
                else:
                    subprocess.call(self.dft_command.split(), stdout=log_fout)
        elif self.energy_driver=='elk':
            with open('log', 'w') as log_fout:
                if self.dft_command is None:
                    subprocess.call(['elk'], stdout=log_fout)
                else:
                    subprocess.call(self.dft_command.split(), stdout=log_fout)
        else:
            raise ValueError('Unknown energy driver specified')
        
    def get_energy(self):
        """
        wraps specific get energy methods for different codes
        """
        if self.energy_driver=='abinit':
            return self._get_energy_abinit()
        elif self.energy_driver=='socorro':
            return self._get_energy_socorro()
        elif self.energy_driver=='elk':
            return self._get_energy_elk()
        else:
            raise ValueError('Unknown energy driver specified')

    def _get_energy_abinit(self):
        """
        reads total energy from abinit log file by finding line with the word etotal.
        If there are multiple etotals reported (maybe in case of sructure relaxation),
        only the final etotal is returned.
        """
        with open('log', 'r') as log_fin:
            for line in log_fin.readlines():
                if ' etotal ' in line:
                    etotal_line = line
        abinit_energy_hartree = float(etotal_line.split()[1])
        return abinit_energy_hartree

    def _get_energy_socorro(self):
        """
        reads total energy from socorro diary file

        Assumes no structure relaxation done in socorro run. If there is
        relaxation, the pre-relaxation energy instead of post-relaxation
        energy is returned and that is bad.

        When Soccoro encounters some error in the solvers (probably due
        to bad input), it still returns success. Therefore, I have to 
        catch this using the fact that the total energy is missing in the
        diaryf. I could instead check that there are no errors in Socorro's
        errorf. 
        """
        # reads and returns energy from socorro output file, diaryf
        with open('diaryf', 'r') as diaryf_fin:
            # iterate reversed so last energy reported is found
            for line in reversed(diaryf_fin.readlines()):
                if 'cell energy   ' in line:
                    soc_energy_rydberg = float(line.split()[3])
                    break
            else:
                raise NoEnergyFromDFT
        soc_energy_hartree = soc_energy_rydberg/2.
        return soc_energy_hartree

    def _get_energy_elk(self):
        elk_energy_all_iterations = np.loadtxt('TOTENERGY.OUT')
        elk_energy_hartree = elk_energy_all_iterations[-1]
        return elk_energy_hartree


class MurnaghanFit(object):
    """
    fits energy vs volume data to murnaghan equation of state

    attributes E0, B0, BP, and V0 are parameters to murnaghan equation derived from fit
    """ 
    def __init__(self, vol_array, E_array):
        self.vol_array = vol_array
        self.E_array = E_array
        murnpars = self._fit_to_murnaghan(self.vol_array, self.E_array)
        self.E0 = murnpars[0]
        self.B0 = murnpars[1]
        self.BP = murnpars[2]
        self.V0 = murnpars[3]

    def _fit_to_murnaghan(self, vol_array, E_array):
        """fts energy vs volume data to murnaghan equation of state"""   
        # fit a parabola to the data to get educated guesses
        a, b, c = np.polyfit(vol_array, E_array, 2)
        # V0 = minimum energy volume, or where dE/dV=0
        # E = aV^2 + bV + c
        # dE/dV = 2aV + b = 0
        # V0 = -b/2a
        # E0 is the minimum energy, which is:
        # E0 = aV0^2 + bV0 + c
        # B is equal to V0*d^2E/dV^2, which is just 2a*V0
        # and from experience we know Bprime_0 is usually a small number like 4
        V0_guess = -b/(2*a)
        E0_guess = a*V0_guess**2. + b*V0_guess + c
        B0_guess = 2.*a*V0_guess
        BP_guess = 4.
        murnpars_guess = [E0_guess, B0_guess, BP_guess, V0_guess]
        murnpars, ier = scipy.optimize.leastsq(self._objective, murnpars_guess, args=(E_array,vol_array))
        return murnpars
    
    def _objective(self,pars,y,x):
        err = y - murnaghan_equation(pars,x)
        return err

def write_murnaghan_data(fit, volumes, abc_list):
    """
    writes fitted murnaghan paramters to file with some useful units
    
    to calculate a, b, and c from volume, the original s and abc_guess are needed
    This assumes isotropic scaling of lattice paramters 
    """
    # calculate miniumum lattice constant
    abc_min = abc_of_vol(fit.V0, volumes[0], abc_list[0])

    # convert results to other units
    abc_min_angstroms = abc_min * 0.52917725
    B0_GPa = fit.B0 * 2 * 1.8218779e-30 / 5.2917725e-11 /4.8377687e-17 / 4.8377687e-17 * 1.e-9

    with open('murnaghan_parameters.dat', 'w') as f:
        f.write('# everything in Hartree atomic units unless specified\n')
        f.write('E_0: %.9f\n' %fit.E0)
        f.write('B_0 (bulk modulus): %.9g\n' %fit.B0)
        f.write('B_0p: %.9f\n' %fit.BP)
        f.write('V_0: %.9f\n' %fit.V0)
        f.write('\n')
        f.write('abc_0, lattice vector scales at minimum energy: %.9f  %.9f  %.9f\n' %tuple(abc_min))
        f.write('\n')
        f.write('B_0 (GPa): %.9f\n' %B0_GPa)
        f.write('abc_0 (angstroms): %.9f  %.9f  %.9f\n' %tuple(abc_min_angstroms))

    
def murnaghan_equation(parameters, vol):
    """murnaghan equation of state"""
    # given iterables of parameters and volumes, return a numpy array of energies.
    # equation From PRB 28,5480 (1983)
    vol = np.array(vol)
    E0 = parameters[0]
    B0 = parameters[1]
    BP = parameters[2]
    V0 = parameters[3]
    E = E0 + B0*vol/BP*(((V0/vol)**BP)/(BP-1)+1) - V0*B0/(BP-1.)
    return E
    

def abc_of_vol(V, V_in, abc_in, two_dim=False):
    """
    calculate a, b, and c given volume of unit cell and one set of known volume, a, b, and c,
    assuming isotropic scaling of a, b, and c
    or, for 2d cases, isotropic scaling of a and b while c is fixed.

    math hint: the ratios between a, b, and c are fixed, 
    meaning b=a*b_guess/a_guess and c=a*c_guess/a_guess. 
    express v in terms of a, V_in, and the guesses, then solve for a.
    """
    ag = float(abc_in[0])
    bg = float(abc_in[1])
    cg = float(abc_in[2])
    if two_dim:
        raise Exception
    else:
        a = np.array(ag*(V/V_in)**(1./3.))
        b = a*bg/ag
        c = a*cg/ag
        return np.array([a,b,c])


class Poly2DFit(object):
    """
    This is used for calculating the equilibrium lattice parameters for materials
    with two independent lattice parameters (eg. hexagonal, wurtzite). Given energy
    calculated on a 2D grid of x and y values, where x and y are the lattice 
    parameters, this calculates a bivariate polynomial fit to the data and finds the x
    and y values where the energy is minimized.
    
    x, y, and z: arrays or lists of same length, x and y are the independent variables
    z is energy

    attributes: 
        coeff: coefficients of the fitted polynomial
        min_xy: x and y where energy is minimized
        min_E: energy at min_xy in Hartree
    """
    def __init__(self, x, y, z):
        x = np.array(x)
        y = np.array(y)
        z = np.array(z) 
        A = np.array([x*0+1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]).T
        coeff, r, rank, s = np.linalg.lstsq(A, z)
        self.coeff = coeff  # coefficients of polynomial
        xy, E = self._find_minimum(x[0], y[0])
        self.min_xy = xy
        self.min_E = E

    def _find_minimum(self, x_guess, y_guess):
        f = lambda xy: poly2d(self.coeff, *xy)
        results = scipy.optimize.minimize(f, (x_guess, y_guess), method='Nelder-Mead')
        return results.x, results.fun


def poly2d(co, x, y):
    """ co contains the 10 coefficents of the cubic bivariate polynomial """
    return co[0] + co[1]*x + co[2]*y + co[3]*x**2 + co[4]*x*y + co[5]*y**2 +co[6]*x**3 + co[7]*x**2*y + co[8]*x*y**2 + co[9]*y**3
    

def write_poly2d_data(fit):
    """
    writes fitted 2D polynomial data to file with some useful units
    includes calculated lattice parameters
    """
    # convert results to other units
    min_xy_angstroms = [x*0.52917725 for x in fit.min_xy]

    with open('poly2d_parameters.dat', 'w') as f:
        f.write('E_0 (Ha): %.9f\n' %fit.min_E)
        f.write('E_0 (Ry): %.9f\n' %(fit.min_E*2.))
        f.write('lattice parameters (Bohr): %.9f  %.9f\n' %tuple(fit.min_xy))
        f.write('lattice parameters (angstroms): %.9f  %.9f\n' %tuple(min_xy_angstroms))
        f.write('\n')
        f.write('polynomial coefficents for ' +
            'c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 +c6*x^3 + c7*x^2*y + c8*x*y**2 + c9*y^3\n')
        f.write('%.9f  %.9f  %.9f  %.9f  %.9f  %.9f  %.9f  %.9f  %.9f  %.9f\n' %tuple(fit.coeff))
        
