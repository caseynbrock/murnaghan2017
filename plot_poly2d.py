import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
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

    # plot data only
    plot_data(a, c, E)
    plt.title('Raw data')
    plt.show(block=False)

    # plot data and fit
    fig, ax = plot_data(a, c, E)
    plt.title('Raw data, fit, and minimum')
    plot_fit(fig, ax, a, c, coeff, minimum=minimum)
    plt.show(block=False)

    raw_input('...')


def plot_data(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('a (bohr)')
    ax.set_ylabel('c (bohr)')
    ax.set_zlabel('energy (Ha)')
    ax.scatter(x,y,z)
    return fig, ax


def plot_fit(fig, ax, x, y, co, minimum=None):
    """
    co: coefficients for fit        
    minimum: local (global?) minumum of polynomial

    should be called after plot_data
    """
    x_fit = np.linspace(x[0], x[-1], 100)
    y_fit = np.linspace(y[0], y[-1], 100)
    X,Y = np.meshgrid(x_fit, y_fit)
    Z = m.poly2d(co, X, Y)
    ax.plot_surface(X, Y, Z, cmap='YlGnBu', alpha=0.7)

    if minimum is not None:
        ax.scatter(minimum[0], minimum[1], m.poly2d(co, minimum[0], minimum[1]), color='r')


if __name__=='__main__':
    main()
