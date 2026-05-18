"""
Set of functions used for 1D plasma, solvers are to be moved to another file 
later on
"""

import numpy as np


def Compute_force(pos, Cell_pot, parameters):
    
    p, d, f, n = parameters["particles"], parameters["domain"], parameters["fft"], parameters["numerical"]

    Part_num = p["Part_num"]
    q = p["q"]

    Cell_num, dx = d["Cell_num"],  d["L"] / d["Cell_num"]
    Cell_pos, Cell_edges = d["Cell_pos"], d["Cell_edges"]
    eps0 = d["eps0"]

    k, k2 = f["k"], f["k2"]

    Relax_step_max, epsilon = n["Relax_step_max"], n["epsilon"]
    solver = n["solver"]

    V = Cell_pot

    Part_per_cell, b = np.histogram(pos, bins=Cell_edges)
    
    rho0 = Part_num / Cell_num                                                 ######## Needs to be cleaned ########

    #rho = rho0 - Part_per_cell                           # charge density
    rho = q*(Part_per_cell - rho0)/dx
    

    # Relaxation scheme #
    if solver == 'p':
        
        for j in range(Relax_step_max):                                        
            Vnew = 0.5 * (np.roll(V,-1) + np.roll(V,1) + dx**2 * (rho / eps0)) # Poisson equation 
        
            delta = np.max(np.abs((Vnew-V)/(V+1e-10)))                         # Convergency check
        
            V = Vnew.copy()
     
            if delta < epsilon :                                               # Optimisation
                break

        Cell_field = -(np.roll(V,-1) - np.roll(V,1)) / (2*dx)                  # Field in each cell
        Cell_pot = V

    
    # Fourier solver #
    else:

        rho_k = np.fft.fft(rho)

        V_k = rho_k / (k2 * eps0)
        V_k[0] = 0.0

        V = np.real(np.fft.ifft(V_k))

        E_k = -1j*k*V_k
        Cell_field = np.real(np.fft.ifft(E_k))  # Field in each cell
        
        Cell_pot = V


    Part_field = np.interp(pos, Cell_pos, Cell_field)             # Field felt by each particle (simple interpolation)

    Part_force = q * Part_field                                   # Force on each particle

    return Part_force, Part_per_cell, Cell_pot, Cell_field




def Euler_exp(pos, vel, dt, force, Cell_pot, parameters):
    p = parameters["particles"]

    m = p["m"]

    dpos_dt = vel
    dvel_dt = force / m

    pos_new = pos + dpos_dt * dt
    vel_new = dpos_dt + dvel_dt * dt

    return pos_new, vel_new, Cell_pot


def Euler_imp(pos, vel, dt, force, Cell_pot, parameters):
    
    p = parameters["particles"]

    m = p["m"]
    
    dpos_dt = vel
    dvel_dt = force / m
    
    vel_new = dpos_dt + dvel_dt * dt
    pos_new = pos + vel_new * dt

    return pos_new, vel_new, Cell_pot


def Leapfrog(pos, vel, dt, force, Cell_pot, parameters):
    
    p, d = parameters["particles"], parameters["domain"]

    m = p["m"]
    L= d["L"]

    dpos_dt = vel
    dvel_dt = force / m
    
    pos_new = pos + dpos_dt * dt + 0.5 * dvel_dt * dt**2
    pos_new = pos_new % L

    force_new, Part_per_cell, Cell_pot, Cell_field = Compute_force(pos_new, Cell_pot, parameters)

    dvel_dt_new = force_new / m

    vel_new = vel + 0.5 * (dvel_dt + dvel_dt_new) * dt

    return pos_new, vel_new, Cell_pot


def RK4(pos, vel, dt, force, Cell_pot, parameters):
    
    p, d = parameters["particles"], parameters["domain"]

    m = p["m"]
    L = d["L"]

    k1_pos = vel
    k1_vel = force / m
    
    pos_temp = (pos + 0.5 * dt * k1_pos) % L
    vel_temp = vel + 0.5 * dt * k1_vel
    force, Part_per_cell, Cell_pot, Cell_field = Compute_force(pos_temp, Cell_pot, parameters)
    k2_pos = vel_temp
    k2_vel = force / m

    pos_temp = (pos + 0.5 * dt * k2_pos) % L
    vel_temp = vel + 0.5 * dt * k2_vel
    force, Part_per_cell, Cell_pot, Cell_field  = Compute_force(pos_temp, Cell_pot, parameters)
    k3_pos = vel_temp
    k3_vel = force / m

    pos_temp = (pos + dt * k3_pos) % L
    vel_temp = vel + dt * k3_vel
    force, Part_per_cell, Cell_pot, Cell_field  = Compute_force(pos_temp, Cell_pot, parameters)
    k4_pos = vel_temp
    k4_vel = force / m

    pos_new = pos + dt * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)/6
    vel_new = vel + dt * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)/6

    return pos_new, vel_new, Cell_pot