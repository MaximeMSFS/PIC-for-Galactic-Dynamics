""" 
This code computes the 1D dynamics of charged particles in a plasma with no 
external field using relaxation of the Poisson equation or Fourrier techniques
with various integration schemes.
"""

import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
from tqdm import tqdm          
init(autoreset=True)

import modules.plotting as plotting
from modules.user_input_grav import User_input
import modules.init as initialize
import modules.schemesCIC_grav as schemes


################################# Parameters ##################################


parameters, main_file, pos_file, vel_file, pot_file, resume_file, save = User_input()


p, d, t, n = parameters["particles"], parameters["domain"], parameters["time"], parameters["numerical"]

m = p["m"]

L, Cell_num, dx = d["L"], d["Cell_num"],  d["L"] / d["Cell_num"]
Cell_pos = d["Cell_pos"]
eps0 = d["eps0"]

Tmin, Tmax, dt0 = t["Tmin"], t["Tmax"], t["dt0"]
variable_dt = n["variable_dt"]
t_arr = t["t"]
initial_step = t["step"]

save_step = n["save_step"]
scheme = n["scheme"]

################################# Initialisation ##############################


Part_pos, Part_vel, Part_charge, Cell_pot, rho0, Part_per_cell = initialize.start(parameters)

plotting.position_plot(Part_pos, parameters)


################################ Main Program #################################
    

dt = dt0
E_tot = []
T_loop = []

if (scheme == 0):
    integrator = schemes.Euler_exp
elif (scheme == 1):
    integrator = schemes.Euler_imp
elif (scheme == 2):
    integrator = schemes.Leapfrog
    print ('leapfrog')
elif (scheme == 3):
    integrator = schemes.RK4
    print ('RK4')
    
    

t_i = Tmin
i = initial_step

progress_bar = tqdm(total=(Tmax-Tmin), desc="sim progress", unit="s", bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}]")

while (t_i < Tmax) :

    Part_force, Part_per_cell, Cell_pot, Cell_field = schemes.Compute_force(Part_pos, Cell_pot, parameters)

    Part_pos, Part_vel, Cell_pot = integrator(Part_pos, Part_vel, dt, Part_force, Cell_pot, parameters)

    Part_pos = Part_pos % L                                            # Periodic BC : % L -> 'modulo L'
    
    # Post processing, validation and other
    if variable_dt == 'yes':
        dt = min(np.min(dx/np.abs(Part_vel+1e-10)), dt0)
    else :
        pass
    
    t_i += dt
    i += 1

    progress_bar.update(dt)
    
    """E_kin = 0.5 * m * np.sum(Part_vel**2)
    E_pot = 0.5 * eps0 * np.sum(Cell_field**2) * dx
    E_tot_i = E_kin + E_pot
    E_tot.append(E_tot_i)"""
    
    T_loop.append(t_i)
    
    if (save == 'yes') and (i % save_step == 0):
        #main_file.write(f"{i:6d} {t_i:10.3f} {E_tot_i:10.3e}\n")
        np.save(pos_file, Part_pos)
        np.save(vel_file, Part_vel)
        np.save(pot_file, Cell_pot)
        
        np.savez(resume_file, step=i, time=t_i, pos=Part_pos, vel=Part_vel, pot=Cell_pot)

        pos_file.flush()
        vel_file.flush()
        pot_file.flush()

    
print (Fore.GREEN + Style.BRIGHT + '\n\nSimulation completed successfully !\n ')
    
if (save == 'yes') :
    main_file.close()
    pos_file.close()
    vel_file.close()
    pot_file.close()
    
    
################################ Plots ########################################


#plotting.conditions_plot(Cell_pos, Part_per_cell, Part_pos, Part_vel, 'Final', Tmax, parameters)

#plotting.field_plot(Cell_pos, Cell_pot, Cell_field, 'Final', Tmax, parameters)

plotting.position_plot(Part_pos, parameters)

plotting.energy_plot(E_tot, T_loop, parameters)

plt.show()