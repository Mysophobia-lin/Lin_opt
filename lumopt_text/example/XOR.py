import os
import sys
import numpy as np
import scipy as sp

from lumopt import CONFIG
from lumopt.geometries.topology import TopologyOptimization2D, TopologyOptimization3DLayered
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths


##

def runSim(params, eps_bg, eps_wg, x_pos, y_pos, z_pos, size_x, size_y, size_z, filter_R, working_dir, beta):

    wavelengths_1 = Wavelengths(start = 1500*1e-9, stop = 1600*1e-9, points = 5)
    
    #geometry = TopologyOptimization2D(params=params, eps_min=eps_min, eps_max=eps_max, x=x_pos, y=y_pos, z=0, filter_R=filter_R, beta=beta)
    
    geometry = TopologyOptimization3DLayered(params=params, eps_min=eps_bg, eps_max=eps_wg, x=x_pos, y=y_pos, z=z_pos, filter_R=filter_R, beta=beta)
    
####FOM###
    fom_1_1 = ModeMatch(monitor_name = 'fom_1', mode_number = 'Fundamental TE mode', direction = 'Forward', target_T_fwd = lambda wl: np.ones(wl.size), norm_p = 2, target_fom=1)
    
    fom_1_0 = ModeMatch(monitor_name = 'fom_1', mode_number = 'Fundamental TE mode', direction = 'Forward', target_T_fwd = lambda wl: np.zeros(wl.size), norm_p = 2, target_fom=0)
    
    
    optimizer = ScipyOptimizers(max_iter=1000, method='L-BFGS-B', pgtol=1e-6, ftol=1e-5, scale_initial_gradient_to=0.25)
    


    script_I3 = load_from_lsf('I_3.lsf')
    script_I3 = script_I3.replace('opt_size_x=10e-6','opt_size_x={:1.6g}'.format(size_x))
    script_I3 = script_I3.replace('opt_size_y=10e-6','opt_size_y={:1.6g}'.format(size_y))


####I_1#####
    
    #opt_I1_1 = Optimization(base_script=script_I1, wavelengths = wavelengths_1, fom=fom_1_1, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False, save_global_index=True)

    #opt_I2_1 = Optimization(base_script=script_I2, wavelengths = wavelengths_1, fom=fom_1_1, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False, save_global_index=True) 

    opt_I3_1 = Optimization(base_script=script_I3, wavelengths = wavelengths_1, fom=fom_1_1, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False, save_global_index=True)



    opt=opt_I3_1#+opt_I1_1
    #opt = opt_I1_1+opt_I2_1+opt_I3_1
    opt.run(working_dir = working_dir)
    
if __name__ == '__main__':
    size_x = 2500
    size_y = 2500
    size_z = 220
    filter_R = 50e-9

    x_points=int(size_x/20)+1
    y_points=int(size_y/20)+1
    z_points=int(size_z/20)+1
    eps_wg = 3.47**2
    eps_bg = 1**2
    x_pos = np.linspace(-size_x/2*1e-9,size_x/2*1e-9,x_points)
    y_pos = np.linspace(-size_y/2*1e-9,size_y/2*1e-9,y_points)
    z_pos = np.linspace(-size_z/2*1e-9,size_z/2*1e-9,z_points)

    ## We need to pick an initial condition. Many different options:
    params = 0.5*np.ones((x_points,y_points))     #< Start with the domain filled with (eps_wg+eps_bg)/2
    #params = np.ones((x_points,y_points))        #< Start with the domain filled with eps_wg
    #params = np.zeros((x_points,y_points))       #< Start with the domain filled with eps_bg    
    #params = None                                #< Use the structure defined in the project file as initial condition

    working_dir = 'XOR_x{:04d}_y{:04d}_f{:04d}'.format(size_x,size_y,int(filter_R*1e9))
    runSim(params, eps_bg, eps_wg, x_pos, y_pos, z_pos, size_x*1e-9, size_y*1e-9, size_z*1e-9, filter_R, working_dir=working_dir, beta=1)



