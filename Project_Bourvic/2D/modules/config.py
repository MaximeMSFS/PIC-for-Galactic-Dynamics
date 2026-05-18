"""
default set of parameters to be modified by the user when starting a new simulation
"""

import numpy as np


# particles #
Part_num = int(1e2) # Number of particles
q = -1              # Charge of particles (C)
m = 1               # Mass of particles (kg)

Pop_A_num = Part_num//2          # Number of particles in population A
Pop_B_num = Part_num - Pop_A_num # Number of particles in population B

pos_disp = 0.1 # Particles position dispertion
Vel_disp = 0.1 # Particles velocity dispertion


# Domain #
L = 1              # Size of the domain (m)
Cell_num = 300     # Number of cells in the domain
dx = L / Cell_num  # Size of a cell (m)

Cell_pos = np.linspace(0, L-dx, Cell_num) + dx/2 # position of each cell (center of the cell)
Cell_pos_x, Cell_pos_y = np.meshgrid(Cell_pos, Cell_pos)

Cell_edges = np.linspace(0, L, Cell_num+1)       # position of each cell borders (used for bins)
Cell_edges_x, Cell_edges_y = np.meshgrid(Cell_edges, Cell_edges)

eps0 = 1.0 # Medium permitivity


# time #
Tmin, Tmax = 0.0, 10.0          # Strating and ending time (s)
dt0 = 1e-3                      # Time step (s)
t = np.arange(Tmin, Tmax, dt0)  # Times array
step = 0                        # Initial step


# FFT wave numbers #
k = 2*np.pi*np.fft.fftfreq(Cell_num, d=dx)
k2 = k**2
k2[0] = 1.0   # avoids division by zero

# numerical #
Relax_step_max = 50
variable_dt = 'no'
epsilon = 1e-4
eps = dx/2           # dumb schmilblick to be removed later on
save_step = 100
random_seed = 0      #for debug, allows for reproductibility



def config():

    config = {}

    config["particles"] = dict(Part_num = Part_num,              
                               Pop_A_num = Pop_A_num,        
                               Pop_B_num = Pop_B_num, 
                               q = q,                        
                               m = m,                           
                               pos_disp = pos_disp,
                               Vel_disp = Vel_disp)

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
                         k2 = k2)

    config["numerical"] = dict(Relax_step_max = Relax_step_max,
                               variable_dt = variable_dt,
                               epsilon = epsilon, 
                               eps = eps, 
                               save_step = save_step,
                               random_seed = random_seed,
                               scheme = 0,
                               solver = 0)

    return config