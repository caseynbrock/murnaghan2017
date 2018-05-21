import murnaghan2017 as m 

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
