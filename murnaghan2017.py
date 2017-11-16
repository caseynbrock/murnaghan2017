#!/bin/env python
import shutil
import os 
import sys
import glob
import subprocess
from scipy.optimize import leastsq
import numpy as np

def main():
    """
    Sets up and runs lattice parameter sweep.

    The generate_lattice_constants function is provided for convenience.
    Alternatively, abc can be created manually. It should be
    an Nx3 numpy array.
    for 2d materials, it is best not to vary layer spacing, so setting 
    two_dim=True will leave lattice paramter c fixed during sweep
    abc = generate_lattice_constants(abc_guess, max_pert, N, two_dim=True)
    """
    # User inputs
    abc_guess = [4, 4, 6]  # lattice parameters guess: a, b, c 
    angles = [90, 90,120]  # lattice vector angles: alpha, beta, gamma
    prim_vec = [[1,1,-1], [-1,1,1], [1,-1,1]]  # lattice vector angles: alpha, beta, gamma
    max_pert = 0.05  # max perturbation of lattice parameters
    N = 7 # number of steps in lattice parameter sweep
    energy_driver = 'abinit'  # dft code to use

    # Set up and run lattice parameter sweep
    abc = generate_lattice_constants(abc_guess, max_pert, N)  # or create abc manually
    raw_energy_data = run_energy_calculations(abc, angles, energy_driver)
    
    write_results(raw_energy_data)

    # Fit energy/volume data to Murnaghan equation of state
    fit_data = murnaghan_fit(raw_energy_data)

    # optionally, plot data
    # maybe_plot_or_something()

class LatticeParameterSweep(object):
    def __init__(self, energy_driver, template_file, abc, angles=None, prim_vec=None, two_dim=False):
        self.energy_driver = energy_driver
        self.template_file = template_file
        self.abc = np.array(abc)
        self.two_dim = two_dim
        self.vol_array = None
        self.E_array = None
        
        # the unit cell can be specified with either primitive vectors (which are scaled by abc)
        # or angles (useful for hex)
        if prim_vec is None and angles is None:
            raise ValueError('Either prim_vec or angles should be defined')
        elif prim_vec is not None and angles is not None:
            raise ValueError('Either prim_vec or angles should be defined, but not both')
        elif prim_vec is not None and angles is None:
            self.prim_vec = np.array(prim_vec)
            self.angles = angles # angles don't need to be calculated
        else:
            self.angles = np.array(angles)
            self.prim_vec = self._prim_vec_from_angles()

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
    
    def _calc_unit_cell_volume(self, s):
        """
        Calculate volume of unit cell from primitive vectors.
        u dot (v cross w) where u,v,w are the prmitive vectors
        u, v, w are s[i]*prim_vec[i]
        """        
        uvw = [s[i]*self.prim_vec[i] for i in range(3)]
        vcrossw = np.cross(uvw[1], uvw[2])
        volume = np.dot(uvw[0],vcrossw)
        return volume


    def preprocess_file(self, s):
        """
        wraps the specific file preprocess functions for different dft codes
        """
        if self.energy_driver=='abinit':
            self._preprocess_file_abinit(s)
        else:
            raise ValueError('Unknown energy driver specified')

    def _preprocess_file_abinit(self, s):
        """
        writes abinit.in from template and appends lattice vector lengths (acell) and 
        EITHER angles (angdeg) or primitive vectors (rprim), depending on which was 
        passed in to the LatticeVectorSweep object
        """
        shutil.copy2(self.template_file, 'abinit.in')
        if self.angles is not None:
            with open('abinit.in', 'a') as f:
                f.write('acell ' + ' '.join([str(float(n)) for n in s]) + '\n')
                f.write('angdeg ' + ' '.join([str(float(n)) for n in self.angles]) + '\n')
        else:
            with open('abinit.in', 'a') as f:
                f.write('acell ' + ' '.join([str(float(n)) for n in s]) + '\n')
                f.write('rprim ' + ' '.join([str(float(n)) for n in self.prim_vec[0]]) + '\n')
                f.write('      ' + ' '.join([str(float(n)) for n in self.prim_vec[1]]) + '\n')
                f.write('      ' + ' '.join([str(float(n)) for n in self.prim_vec[2]]) + '\n')

    def run_energy_calculations(self):
        """
        Uses DFT code to calculate energy at each of the lattice constants specified.
    
        For each lattice constant in sweep, sets up directory and runs dft code in that directory.
        """  
        # remove old work directories
        for f in glob.glob('workdir.*'):
            shutil.rmtree(f)
        
        # run dft calculations and return calculated energies
        main_dir = os.getcwd()
        energy_list_hartree = []
        for i,s in enumerate(self.abc):
            dir_name = 'workdir.'+str(i)
            shutil.copytree('templatedir', dir_name)
            os.chdir(dir_name)
            self.preprocess_file(s)
            self.run_dft()
            energy_list_hartree.append(self.get_energy())
            os.chdir(main_dir)
        return energy_list_hartree


    def run_dft(self):
        """
        runs dft code in current directory
        """
        if self.energy_driver=='abinit':
            with open('log', 'w') as log_fout, open('files','r') as files_fin:
                #subprocess.call(['srun', '-n', '64', 'abinit'], stdin=files_fin, stdout=log_fout)
                subprocess.call(['abinit'], stdin=files_fin, stdout=log_fout)
        else:
            raise ValueError('Unknown energy driver specified')
        
        
    def get_energy(self):
        """
        wraps specific get energy methods for different codes
        """
        if self.energy_driver=='abinit':
            return self._get_energy_abinit()
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
        abinit_energy_hartree = np.float(etotal_line.split()[1])
        return abinit_energy_hartree
    
    def run_abinit():
        #call abinit
        with open('log', 'w') as log_fout, open('files','r') as files_fin:
            #subprocess.call(['srun', '-n', '64', 'abinit'], stdin=files_fin, stdout=log_fout)
            subprocess.call(['abinit'], stdin=files_fin, stdout=log_fout)
        # read energy from log
        energy = abinit_get_energy()
    
    def fit_sweep_to_murnaghan(self):
        """wrapper around fit_to_murnaghan function that passes volume and energy data from this sweep"""      
        if self.vol_array is None:
            raise ValueError('No volume data!')
        if self.E_array is None:
            raise ValueError('No energy data!')
        return MurnaghanFit(self.vol_array, self.E_array)

    def write_energy_data(self):
        """ 
        writes raw energy data from dft runs.
        The header should contain primitive vectors, and angles if specified
        The tabulated data should have 
        a, b, c, L_a, B, C, V, E_Ha, E_Ry, E_eV for each run
        where a,b,c are scaling factors for primitve vectors (bohr)
        L_a, L_b, L_c, are the lenghts of primitve vectors (same as a,b,c in some cases) (bohr)
        V is the volume of the unit cell (Bohr^3)
        E_Ha is the total energy in Hartree,
        E_eV is the total energy in eV
        """
        pass

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
        murnpars, ier = leastsq(self._objective, murnpars_guess, args=(E_array,vol_array))
        return murnpars
    
    def _objective(self,pars,y,x):
        err = y -  murnaghan_equation(pars,x)
        return err


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
    

def generate_lattice_constants(abc_guess, max_pert, N, two_dim=False):
    """
    generates an Nx3 numpy array containing the three lattice constants for each energy calculation. 
    Each value will range from (1-max_pert)*abc_guess[i] to (1+max_pert)*abc_guess[i].

    two_dim: if true, will only scale abc_guess[0] and abc_guess[1].
        Useful for 2d materials where layer spacing should remain constant.
    abc_guess: guess of lattice constants a, b, and c (list or numpy array)
    max_pert: the amount to perturb the lattice constants, eg. 0.05 means +-5%
    N: number of steps 
    """
    abc_guess = np.array(abc_guess)
    pert_list = np.linspace(1-max_pert, 1+max_pert, N)
    abc = np.array([pert*abc_guess for pert in pert_list])
    if two_dim:
        abc[:,2] = abc_guess[2] # third lattice parameter not varied
    return abc


