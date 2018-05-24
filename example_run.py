import murnaghan2017 as m 
import numpy as np


# EXAMPLE RUN FOR BCC STRUCTURE USING MURNAGHAN EOS AND
# THE ABINIT CODE.
# THE TEMPLATEDIR DIRECTORY WITH THE TEMPLATE FILE
# AND PSEUDOPOTENTIALS/PAWs NEEDS TO BE SET UP AS DESCRIBED
# IN THE README 

energy_driver = 'abinit'
# template file is usually 'abinit.in.template' for abinit,
# 'crystal.template' for socorro,
# or 'elk.in.template' for elk
template_file = 'abinit.in.template'
# scales to loop though. These get multiplied by abc_guess
s = [0.95, 0.975, 1.0,  1.025, 1.05]
# lattice parameter guesses
abc_guess = [6.3, 6.3, 6.3]
# this list of abc arrays gets passed to actual sweep
abc_list = [si*np.array(abc_guess) for si in s]
# unscaled primitve vectors
pvu = [[ 0.5,  0.5, -0.5],
       [-0.5,  0.5,  0.5],
       [ 0.5, -0.5,  0.5]]
# run lattice paramter sweep
volumes, energy_list_hartree = m.lattice_parameter_sweep(energy_driver, template_file, abc_list, prim_vec_unscaled=pvu)
# fit to EOS and write fit data to file
fit = m.MurnaghanFit(volumes, energy_list_hartree)
m.write_murnaghan_data(fit, volumes, abc_list)



# # EXAMPLE RUN FOR WURTZITE GaN USING 2D POLYNOMIAL FIT
# # AND THE SOCORRO CODE.
# # THE TEMPLATEDIR DIRECTORY WITH THE CRYSTAL TEMPLATE
# # AND PSEUDOPOTENTIALS/PAWs NEEDS TO BE SET UP AS DESCRIBED
# # IN THE README 
#
# energy_driver = 'socorro'
# # template file is usually 'abinit.in.template' for abinit, 
# # 'crystal.template' for socorro, 
# # or 'elk.in.template' for elk
# template_file = 'crystal.template'
# 
# # create list of a,b,c values to test
# s_a = [0.99, 0.995, 1.0, 1.005, 1.01]  # scales for a/b lattice parameters
# s_c = [0.99, 0.995, 1.0, 1.005, 1.01]  # scales for c lattice parameter
# 
# abc_guess = np.array([5.97, 5.97, 9.7])
# abc_list = []
# for s1 in s_a:
#     for s2 in s_c:
#         abc_list.append(np.array([s1, s1, s2]) * abc_guess)
# abc_list = np.array(abc_list)
# 
# # unscaled primitive vectors
# angle = 120.*np.pi/180.
# pvu = [[1.0,  0.0,  0.0],
#        [np.cos(angle),  np.sin(angle),  0.0],
#        [0.0,  0.0,  1.0]]
# # run lattice parameter sweep
# volumes, energy_list_hartree = m.lattice_parameter_sweep(energy_driver, 
#     template_file, abc_list, prim_vec_unscaled=pvu)
# 
# # Fit E(a,c) data to 2d polynomial
# a_values = abc_list[:,0]
# c_values = abc_list[:,2]
# fit = m.Poly2DFit(a_values, c_values, energy_list_hartree)
# m.write_poly2d_data(fit)
