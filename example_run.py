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
# unscaled primitve vectors
pvu = [[ 0.5,  0.5, -0.5], 
       [-0.5,  0.5,  0.5], 
       [ 0.5, -0.5,  0.5]]
# run lattice paramter sweep
sweep = m.LatticeParameterSweep(energy_driver, template_file, s, abc_guess, prim_vec_unscaled=pvu)
sweep.run_energy_calculations()
