import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import scipy.optimize
import murnaghan2017 as m


def main():
    """
    Reads in energy data and polynomial fit parameters and passes to plot 
    funtion.

    Currently assumes that a and c are the lattice parameters, but this
    could be changed or generalized.
    """
    # read data from energies.dat and poly2d_parameters.dat
    energy_volume_data = np.loadtxt('energies.dat')
    a    = energy_volume_data[:,0]
    b    = energy_volume_data[:,1]
    c    = energy_volume_data[:,2]
    vols = energy_volume_data[:,3]
    E    = energy_volume_data[:,4]  # Hartree

    with open('poly2d_parameters.dat') as poly2d_file:
        printed_results = poly2d_file.readlines()
        for i in range(len(printed_results)):
            line = printed_results[i]
            if 'E_0 (Ha):' in line:
                E0 = float(line.split()[-1])  # not used?
            if 'lattice parameters (Bohr):' in line:
                minimum = np.array(map(float, line.split()[-2:]))
            if 'polynomial coefficents' in line:
                next_line = printed_results[i+1]
                coeff = np.array(map(float, next_line.split()))

    plot_fit(a, c, E, coeff, minimum=minimum)


# class Poly2DFit(object):
#     """
#     z should be a 1D array with length len(x)*len(y)
#     """
#     def __init__(self, x, y, z):
#         A = np.array([x*0+1, x, y, x**2, x**2*y, x**2*y**2, y**2, x*y**2, x*y]).T
#         coeff, r, rank, s = np.linalg.lstsq(A, z)
#         self.coeff = coeff  # coefficients of polynomial
#         xy, E = self._find_minimum(x[0], y[0])
#         self.min_xy = xy
#         self.min_E = E
#         
#     def _poly2d(self, co, X, Y):
#         return co[0] + co[1]*X + co[2]*Y + co[3]*X**2 + co[4]*X**2*Y + co[5]*X**2*Y**2 +co[6]*Y**2 + co[7]*X*Y**2 + co[8]*X*Y
#     
#     def _find_minimum(self, x_guess, y_guess):
#         f = lambda xy: self._poly2d(self.coeff, *xy)
#         results = scipy.optimize.minimize(f, (x_guess, y_guess), method='Nelder-Mead')
#         return results.x, results.fun
#         
#     def write_results():
#         """ write everything necessary for plot script """
#         pass


def plot_fit(x, y, z, co, minimum=None):
    """
    co is coefficients for fit
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('a (bohr)')
    ax.set_ylabel('c (bohr)')
    ax.set_zlabel('energy (Ha)')
    ax.scatter(x,y,z)

    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = np.linspace(y[0], y[-1], 100)
    X,Y = np.meshgrid(x_fit, y_fit)
    Z = m.poly2d(co, X, Y)
    ax.plot_surface(X, Y, Z, cmap='YlGnBu', alpha=0.7)

    if minimum is not None:
        ax.scatter(minimum[0], minimum[1], m.poly2d(co, minimum[0], minimum[1]), color='r')

    plt.show()
    

if __name__=='__main__':
    main()
