"""
default set of parameters to be modified by the user when starting a new simulation
"""

import numpy as np


# particles #
Part_num = int(1e5)  # Number of particles
q = 0                # Charge of particles (C)
m = 1/Part_num              # Mass of particles (kg)

Pop_A_num = Part_num//2           # Number of particles in population A
Pop_B_num = Part_num - Pop_A_num  # Number of particles in population B

pos_disp = 0.1  # Particles position dispertion (m)
Vel_disp = 0.1  # Particles velocity dispertion (m.s-1)

omega = 0.1          # rotational term for velocity initialization
gamma = 0.1          # adiabatic constant


# Domain #
L = 1              # Size of the domain (m)
Cell_num = 124     # Number of cells in the domain
dx = L / Cell_num  # Size of a cell (m)

Cell_pos = np.linspace(0, L-dx, Cell_num) + dx/2                               # position of each cell (center of the cell)
Cell_pos_x, Cell_pos_y = np.meshgrid(Cell_pos, Cell_pos, indexing='ij')

Cell_edges = np.linspace(0, L, Cell_num+1)                                     # position of each cell borders (used for bins)
Cell_edges_x, Cell_edges_y = np.meshgrid(Cell_edges, Cell_edges, indexing='ij')

eps0 = 1.0 # Medium permitivity


# time #
Tmin, Tmax = 0.0, 100.0         # Strating and ending time (s)
dt0 = 1e-3                      # Time step (s)
t = np.arange(Tmin, Tmax, dt0)  # Times array
step = 0                        # Initial step (non zero if computation is resumed)


# FFT wave numbers #
k = 2*np.pi*np.fft.fftfreq(Cell_num, d=dx)
kx, ky = np.meshgrid(k, k, indexing='ij')
k2 = kx**2 + ky**2                         # optimization, avoids recomputing everytime
k2[0,0] = 1.0                              # stability, avoids division by zero

# numerical #
Relax_step_max = 50  # optimization, max relax step for the Poisson solver
variable_dt = ''     # is variable time step used by default (yes/no), overwritten by User_Input later on
epsilon = 1e-4       # optimization, convergency check for the Poisson solver
eps = dx/2           # dumb schmilblick to be removed later on
save_step = 100      # save every other step
random_seed = 0      # debug, allows for reproductibility



def config():

    config = {}

    config["particles"] = dict(Part_num = Part_num,              
                               Pop_A_num = Pop_A_num,        
                               Pop_B_num = Pop_B_num, 
                               q = q,                        
                               m = m,                           
                               pos_disp = pos_disp,
                               Vel_disp = Vel_disp,
                               omega = omega,
                               gamma = gamma)

    config["domain"] = dict(L = L,
                            Cell_num = Cell_num,
                            dx = dx,
                            Cell_pos = Cell_pos,
                            Cell_edges = Cell_edges,        
                            eps0 = eps0)     
    
    config["time"] = dict(Tmin = Tmin,             
                          Tmax = Tmax,                
                          dt0 = dt0,
                          t = t,
                          step = step)
    
    config["fft"] = dict(k = k, 
                         k2 = k2,
                         kx = kx,
                         ky = ky)

    config["numerical"] = dict(Relax_step_max = Relax_step_max,
                               variable_dt = variable_dt,
                               epsilon = epsilon, 
                               eps = eps, 
                               save_step = save_step,
                               random_seed = random_seed,
                               scheme = 0,
                               solver = 0)

    return config