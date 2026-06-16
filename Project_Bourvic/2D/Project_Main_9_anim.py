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
import modules.init_grav as initialize
import modules.schemesCIC_grav as schemes
import modules.checks_grav as check

import os

frame_dir = "frames"
os.makedirs(frame_dir, exist_ok=True)

frame_id = 0
save_every = 20


################################# Parameters ##################################


parameters, main_file, pos_file, vel_file, pot_file, resume_file, save = User_input()


p, d, t, n = parameters["particles"], parameters["domain"], parameters["time"], parameters["numerical"]

Pop_A_num, Pop_B_num = p["Pop_A_num"], p["Pop_B_num"]
Pop_num = Pop_A_num + Pop_B_num
gamma = p['gamma']

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
Lz_tot = []
T_loop = []


if (scheme == 0):
    integrator = schemes.Euler_exp
elif (scheme == 1):
    integrator = schemes.Euler_imp
elif (scheme == 2):
    integrator = schemes.Leapfrog
elif (scheme == 3):
    integrator = schemes.RK4
    
    

t_i = Tmin
i = initial_step

progress_bar = tqdm(total=(Tmax-Tmin), desc="sim progress", unit="s", bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]")


###############################################################################

plot_every = 20

plt.ion()

fig, ax = plt.subplots(1,1, figsize=(10,10), dpi=100)

phase_plot_A = ax.scatter(Part_pos[:Pop_A_num,0], Part_pos[:Pop_A_num,1], s=0.3, color='blue', alpha=0.5, label='Population A')

phase_plot_B = ax.scatter(Part_pos[Pop_A_num:,0], Part_pos[Pop_A_num:,1], s=0.3, color='red', alpha=0.5, label='Population B')

phase_plot_center, = ax.plot( L/2, L/2, '+', color='k', alpha=1.0, markersize=5)

criterion_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)

ax.set_xlim(0,L)
ax.set_ylim(0,L)
ax.set_xlabel("x Position (m)")
ax.set_ylabel("y Position (m)")
ax.set_title("2D Dynamics of a Gaz Cloud", fontweight='bold')
ax.grid(alpha=0.2)

fig1, ax1 = plt.subplots(1,1)

rot_curve, = ax1.plot([], [], label='angular velocity')
rot_disp, = ax1.plot([], [], '--', label='angular velocity dispersion')

ax1.grid()
ax1.set_xlim(0,L/2)
ax1.set_ylim(0,1)
ax1.set_xlabel("Distance to the center of rotation (m)")
ax1.set_ylabel("velocity (m.s-1)")
ax1.set_title("Rotation Curves", fontweight='bold')
plt.legend()


fig2, ax2 = plt.subplots()

toomre_plot = ax2.imshow(np.zeros((Cell_num,Cell_num)), vmin=0, vmax=1e-24, cmap='magma', animated=True)
plt.colorbar(toomre_plot)


###############################################################################
x0, y0 = L/2, L/2

while (t_i < Tmax) :

    Part_force, Part_per_cell, Cell_pot, Cell_field = schemes.Compute_force(Part_pos, Cell_pot, parameters)

    Part_pos, Part_vel, Cell_pot = integrator(Part_pos, Part_vel, dt, Part_force, Cell_pot, parameters)
    
    
    dx = (Part_pos[:,0] - x0 + L/2) % L - L/2
    dy = (Part_pos[:,1] - y0 + L/2) % L - L/2
    
    r = np.sqrt(dx**2 + dy**2) + 1e-6
    
    vr = (Part_vel[:,0]*dx + Part_vel[:,1]*dy) / r

    Part_vel[:,0] -= gamma * vr * dx/r * dt
    Part_vel[:,1] -= gamma * vr * dy/r * dt

    Part_pos = Part_pos % L                                            # Periodic BC : % L -> 'modulo L'
    
    # Post processing, validation and other
    if variable_dt == 'yes':
        dt = min(np.min(dx/np.abs(Part_vel+1e-10)), dt0)
    else :
        pass
    
    t_i += dt
    i += 1

    progress_bar.update(dt)
    
    E_tot_i = check.energy(Part_vel, Part_per_cell, Cell_pot, parameters)
    E_tot.append(E_tot_i)
    
    Lz_i = check.angular_momentum(Part_pos, Part_vel, parameters)
    Lz_tot.append(Lz_i)
    
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
        

    
    
    if i % plot_every == 0:
        
        x = Part_pos[:,0]
        y = Part_pos[:,1]
            
        x_c = np.angle(np.mean(np.exp(1j*2*np.pi*x/L))) * L/(2*np.pi)
        y_c = np.angle(np.mean(np.exp(1j*2*np.pi*y/L))) * L/(2*np.pi)
            
        x_shift = (x - x_c + L/2) % L
        y_shift = (y - y_c + L/2) % L

        phase_plot_A.set_offsets(
            np.column_stack(( x_shift[:Pop_A_num],  y_shift[:Pop_A_num])))

        phase_plot_B.set_offsets(
            np.column_stack(( x_shift[Pop_A_num:],  y_shift[Pop_A_num:])))
        
        phase_plot_center.set_data([x_c%L],  [y_c%L])
        
        criterion = check.ostriker_peebles(Part_pos, Part_vel, Part_per_cell, Cell_pot, parameters)
        criterion_text.set_text(f"Ostriker-Peebles Criterion: {criterion:.4f}")
        
        bcen, rot_vel, sigma = check.rotation_curve(Part_pos, Part_vel, parameters)
        rot_curve.set_data(bcen, rot_vel)
        rot_disp.set_data(bcen, sigma)
        ax1.set_ylim(0,np.max([np.max(rot_vel), np.max(sigma)]) * 1.25)
        
    if i % 60 == 0:
        Q, u, v = check.Toomre(Part_pos, Part_vel, parameters, nb=32)
        
        toomre_plot.set_data(Q)
    
    if i % save_every == 0:
        fig.savefig(
            f"{frame_dir}/frame_{frame_id:05d}.jpg",
            dpi=100,              # lower = lighter
            bbox_inches='tight'
        )
        frame_id += 1
        plt.pause(1/200)


    
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