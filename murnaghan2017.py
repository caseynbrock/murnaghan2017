#!/bin/env python
import shutil
import os 
import sys
import glob
import subprocess
from scipy.optimize import leastsq
import numpy as np


class LatticeParameterSweep(object):
    """
    abc_guess = guess for a, b, c (list or array, length 3)
    s: scales (list, length N>4)
    prim_vec_unscaled: unscaled lattice vectors (3x3 array or list of lists)
    angles: angles betwen lattice vectors, alpha, beta, gamma 

    Either prim_vec_unscaled or angles needs to be input, but not both. 
    If angles is defined, prim_vec_unscaled is calculated from angles and all vectors have length 1.

    The actual primitive vectors for a single dft run are s*abc_guess[i]*prim_vec_unscaled[i],  i=1,2,3

    need N > 4 because fitting to murnaghan equation

    angles, prim_vec_unscaled, and s are converted to floats for safety

    """
    def __init__(self, energy_driver, template_file, s, abc_guess, angles=None, prim_vec_unscaled=None, two_dim=False):
        self.energy_driver = energy_driver
        self.template_file = template_file
        self.s = map(float, s)
        self.abc_guess = np.array(map(float, abc_guess))
        self.two_dim = two_dim
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

        self.acell = [s_i*self.abc_guess for s_i in s]
        # Initialize instance attributes:
        self.volumes = None 
        self.energies_hartree = None
        self.murnaghan_fit = None

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
    
    def _calc_unit_cell_volume(self, ind):
        """
        Calculate volume of unit cell from primitive vectors.
        (a*b*c) * u.(vxw) where u,v,w are the unscaled prmitive vectors
        and a,b,c are acell[ind]
 
        ind is index of s to use
        """        
        a = self.acell[ind][0]
        b = self.acell[ind][1]
        c = self.acell[ind][2]
        u = self.prim_vec_unscaled[0]
        v = self.prim_vec_unscaled[1]
        w = self.prim_vec_unscaled[2]
        V_unscaled = np.dot(u, np.cross(v,w))
        V = a*b*c*V_unscaled
        return V


    def _preprocess_file(self, ind):
        """
        wraps the specific file preprocess functions for different dft codes
        """
        if self.energy_driver=='abinit':
            self._preprocess_file_abinit(ind)
        elif self.energy_driver=='socorro':
            self._preprocess_file_socorro(ind)
        elif self.energy_driver=='elk':
            self._preprocess_file_elk(ind)
        else:
            raise ValueError('Unknown energy driver specified')

    def _preprocess_file_abinit(self, ind):
        """
        writes abinit.in from template and appends lattice vector lengths (acell) and 
        EITHER angles (angdeg) or primitive vectors (rprim), depending on which was 
        passed in to the LatticeVectorSweep object
        """
        shutil.copy2(self.template_file, 'abinit.in')
        if self.angles is not None:
            with open('abinit.in', 'a') as f:
                f.write('acell ' + ' '.join([str(float(n)) for n in self.acell[ind]]) + '\n')
                f.write('angdeg ' + ' '.join([str(float(n)) for n in self.angles]) + '\n')
        else:
            with open('abinit.in', 'a') as f:
                f.write('acell ' + ' '.join([str(float(n)) for n in self.acell[ind]]) + '\n')
                f.write('rprim ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[0]]) + '\n')
                f.write('      ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[1]]) + '\n')
                f.write('      ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[2]]) + '\n')

    def _preprocess_file_socorro(self, ind):
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
            this_acell = self.acell[ind]
            scale = this_acell[0]
            prim_vec_0 = self.prim_vec_unscaled[0]
            prim_vec_1 = this_acell[1]/this_acell[0] * self.prim_vec_unscaled[1]
            prim_vec_2 = this_acell[2]/this_acell[0] * self.prim_vec_unscaled[2]
            with open('crystal.template', 'r') as fin:
                template = fin.readlines()
            template.insert(1, '  '+str(scale)+'\n')
            template.insert(2, '    ' + ' '.join(map(str, prim_vec_0)) + '\n')
            template.insert(3, '    ' + ' '.join(map(str, prim_vec_1)) + '\n')
            template.insert(4, '    ' + ' '.join(map(str, prim_vec_2)) + '\n')
            with open('crystal', 'w') as fout:
                for line in template:
                    fout.write(line)

    def _preprocess_file_elk(self, ind):
        """
        writes elk.in from template and appends lattice vector lengths (scale1, scale2, scale3) and 
        primitive vectors (avec)
        """
        shutil.copy2(self.template_file, 'elk.in')
        if self.angles is not None:
            raise ValueError('angdeg input not implemented for elk yet')
        else:
            with open('elk.in', 'a') as f:
                scale123 = self.acell[ind]
                f.write('\nscale1\n  ' + str(float(scale123[0])) + '\n')
                f.write('\nscale2\n  ' + str(float(scale123[1])) + '\n')
                f.write('\nscale3\n  ' + str(float(scale123[2])) + '\n')
                f.write('\navec\n') 
                f.write('  ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[0]]) + '\n')
                f.write('  ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[1]]) + '\n')
                f.write('  ' + ' '.join([str(float(n)) for n in self.prim_vec_unscaled[2]]) + '\n')


    def run_energy_calculations(self):
        """
        Uses DFT code to calculate energy at each of the lattice constants specified.
    
        For each lattice constant in sweep, sets up directory and runs dft code in that directory.
        this method sets instance attirbutes: volumes, energies_hartree, murnaghan_fit
        """  
        # remove old work directories
        for f in glob.glob('workdir.*'):
            shutil.rmtree(f)
        
        # run dft calculations and return calculated energies
        main_dir = os.getcwd()
        energy_list_hartree = []
        for i in range(len(self.s)):
            dir_name = 'workdir.'+str(i)
            self._setup_workdir(dir_name)
            os.chdir(dir_name)
            self._preprocess_file(i)
            self.run_dft()
            energy_list_hartree.append(self.get_energy())
            os.chdir(main_dir)
        # set instance variables
        self.volumes = np.array([self._calc_unit_cell_volume(ind) for ind in range(len(self.s))])
        self.energies_hartree = np.array(energy_list_hartree)
        self.murnaghan_fit = self._fit_sweep_to_murnaghan()
        
        # write raw data and murnaghan fit data to files
        self._write_energy_data()
        self._write_murnaghan_data()

    def _setup_workdir(self, dir_name):
        if self.energy_driver=='abinit':
            shutil.copytree('templatedir', dir_name)
        elif self.energy_driver=='socorro':
            shutil.copytree('templatedir', dir_name)
            this_dir = os.getcwd()
            os.chdir(dir_name)
            os.mkdir('data')
            os.chdir('data')
            os.symlink(os.path.join('..', 'crystal'), 'crystal')
            for file in [os.path.basename(x) for x in glob.glob('../PAW.*')]:
                os.symlink(os.path.join('..', file), file)
            os.chdir(this_dir)
        elif self.energy_driver=='elk':
            shutil.copytree('templatedir', dir_name)
        else:
            raise ValueError('Unknown energy driver specified')

    def run_dft(self):
        """
        runs dft code in current directory
        """
        if self.energy_driver=='abinit':
            with open('log', 'w') as log_fout, open('files','r') as files_fin:
                #subprocess.call(['srun', '-n', '64', 'abinit'], stdin=files_fin, stdout=log_fout)
                subprocess.call(['abinit'], stdin=files_fin, stdout=log_fout)
        elif self.energy_driver=='socorro':
            with open('log', 'w') as log_fout:
                subprocess.call(['socorro'], stdout=log_fout)
        elif self.energy_driver=='elk':
            with open('log', 'w') as log_fout:
                subprocess.call(['elk'], stdout=log_fout)
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
        """
        # reads and returns energy from socorro output file, diaryf
        with open('diaryf', 'r') as diaryf_fin:
            for line in diaryf_fin.readlines():
                if 'cell energy   ' in line:
                    soc_energy_rydberg = float(line.split()[3])
        soc_energy_hartree = soc_energy_rydberg/2.
        return soc_energy_hartree

    def _get_energy_elk(self):
        elk_energy_all_iterations = np.loadtxt('TOTENERGY.OUT')
        elk_energy_hartree = elk_energy_all_iterations[-1]
        return elk_energy_hartree
    
    # def run_abinit():
    #     #call abinit
    #     with open('log', 'w') as log_fout, open('files','r') as files_fin:
    #         #subprocess.call(['srun', '-n', '64', 'abinit'], stdin=files_fin, stdout=log_fout)
    #         subprocess.call(['abinit'], stdin=files_fin, stdout=log_fout)
    #     # read energy from log
    #     energy = abinit_get_energy()
    
    def _fit_sweep_to_murnaghan(self):
        """wrapper around fit_to_murnaghan function that passes volume and energy data from this sweep"""      
        if len(self.s) < 4:
            raise ValueError('Murnaghan equation hs 4 parameters...need more points in abc')
        if self.volumes is None:
            raise ValueError('No volume data!')
        if self.energies_hartree is None:
            raise ValueError('No energy data!')
        return MurnaghanFit(self.volumes, self.energies_hartree)

    def _write_murnaghan_data(self):
        """
        writes fitted murnaghan paramters to file with some useful units
        
        to calculate a, b, and c from volume, the original s and abc_guess are needed
        This assumes isotropic scaling of lattice paramters 
        """
        fit = self.murnaghan_fit 
        # calculate miniumum lattice constant
        abc_min = abc_of_vol(fit.V0, self.volumes[0], self.acell[0])

        # convert results to other units
        abc_min_angstroms = abc_min * 0.52917725
        B0_gpa = fit.B0 * 1.8218779e-30 / 5.2917725e-11 / \
                  4.8377687e-17 / 4.8377687e-17 * 1.e-9
    
        with open('murnaghan_parameters.dat', 'w') as f:
            f.write('# everything in Hartree atomic units unless specified\n')
            f.write('E_0: %.9f\n' %fit.E0)
            f.write('B_0 (bulk modulus): %.9g\n' %fit.B0)
            f.write('B_0p: %.9f\n' %fit.BP)
            f.write('V_0: %.9f\n' %fit.V0)
            f.write('\n')
            f.write('abc_0, lattice vector scales at minimum energy: %.9f  %.9f  %.9f\n' %tuple(abc_min))
            f.write('\n')
            f.write('B_0 (GPa): %.9f\n' %B0_gpa)
            f.write('abc_0 (angstroms): %.9f  %.9f  %.9f\n' %tuple(abc_min_angstroms))


    def _write_energy_data(self):
        """ 
        Writes raw energy data from dft runs.

        The header should contain primitive vectors. 
        where a,b,c are scaling factors for primitve vectors (bohr)
        V is the volume of the unit cell (Bohr^3)
        E_Ha is the total energy in Hartree
        E_eV is the total energy in eV
        """
        with open('energies.dat', 'w') as fout:
            fout.write('# Note that a, b, and c are scales and need to be multiplied \n'+
                       '# by unscaled primitive vectors prim_vec to get actual unit cell vectors\n')
            fout.write('# unscaled primitive vectors:\n')
            fout.write('# '+str(self.prim_vec_unscaled[0]) + '\n')
            fout.write('# '+str(self.prim_vec_unscaled[1]) + '\n')
            fout.write('# '+str(self.prim_vec_unscaled[2]) + '\n')
            fout.write('# a (bohr),   b (bohr),   c (bohr),   V (bohr^3), E (Ha),     E(eV)\n')
            for i in range(len(self.s)):
                a = self.acell[i][0]
                b = self.acell[i][1]
                c = self.acell[i][2]
                V = self.volumes[i] 
                E_Ha = self.energies_hartree[i]
                E_eV = E_Ha*27.21138602
                fout.write('%.9f   %.9f   %.9f   %.9f   %.9f   %.9f \n'
                        %(a, b, c, V, E_Ha, E_eV))



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




