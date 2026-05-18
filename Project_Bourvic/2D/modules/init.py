"""
initialization functions depending on wether or not the user wants to resume a
previous calculation
"""
import numpy as np

def start(parameters):

    p, d, n = parameters["particles"], parameters["domain"], parameters["numerical"]

    Part_num, Pop_A_num, Pop_B_num = p["Part_num"], p["Pop_A_num"], p["Pop_B_num"]
    q = p["q"]
    pos_disp, Vel_disp = p["pos_disp"], p["Vel_disp"]

    L, Cell_num = d["L"], d["Cell_num"]
    Cell_edges = d["Cell_edges"]

    random_seed = n["random_seed"]


    np.random.seed(random_seed)    #for debug, allows for reproductibility

    Part_pos = np.zeros((Part_num,2))
    Part_vel = np.zeros((Part_num,2))

    Part_pos[0:Pop_A_num,0] = L*np.random.rand(Pop_A_num)
    Part_pos[Pop_A_num:,0] = L*np.random.rand(Pop_B_num)
    
    Part_pos[0:Pop_A_num,1] = L*np.random.rand(Pop_A_num)
    Part_pos[Pop_A_num:,1] = L*np.random.rand(Pop_B_num)

    Part_vel[0:Pop_A_num,0] = 1.0 - Vel_disp/2 + Vel_disp * np.random.rand(Pop_A_num)
    Part_vel[Pop_A_num:,0] = -1.0 - Vel_disp/2 + Vel_disp * np.random.rand(Pop_B_num)
    
    Part_vel[0:Pop_A_num,1] = 1.0 - Vel_disp/2 + Vel_disp * np.random.rand(Pop_A_num)
    Part_vel[Pop_A_num:,1] = -1.0 - Vel_disp/2 + Vel_disp * np.random.rand(Pop_B_num)
    
    Part_vel[:,0] += 1e-3 * np.random.randn(*Part_vel[:,0].shape)

    Part_charge = np.ones(Part_num) * q

    Cell_pot = np.zeros(Cell_num)

    #one particle type
    rho0 = Part_num / Cell_num # Intrinsic charge density

    #test of initial conditions
    Part_per_cell, b = np.histogram(Part_pos, bins=Cell_edges)
    
    return Part_pos, Part_vel, Part_charge, Cell_pot, rho0, Part_per_cell