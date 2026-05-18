"""
initialization functions depending on wether or not the user wants to resume a
previous calculation
"""
import numpy as np

def start(parameters):

    
    # ========================================================================
    
    
    p, d, n = parameters["particles"], parameters["domain"], parameters["numerical"]

    Part_num, Pop_A_num, Pop_B_num = p["Part_num"], p["Pop_A_num"], p["Pop_B_num"]
    m = p["m"]
    omega = p["omega"]
    
    pos_disp, Vel_disp = p["pos_disp"], p["Vel_disp"]

    L, Cell_num = d["L"], d["Cell_num"]
    Cell_edges = d["Cell_edges"]

    random_seed = n["random_seed"]
    
    
    # ========================================================================


    np.random.seed(random_seed)    # debug, allows for reproductibility
    
    Part_pos = np.zeros((Part_num,2))
    Part_vel = np.zeros((Part_num,2))
    
    
    # positions #
    # x 
    Part_pos[0:Pop_A_num,0] = L*np.random.rand(Pop_A_num)
    Part_pos[Pop_A_num:,0] = L*np.random.rand(Pop_B_num)
    
    # y
    Part_pos[0:Pop_A_num,1] = L*np.random.rand(Pop_A_num)
    Part_pos[Pop_A_num:,1] = L*np.random.rand(Pop_B_num)
    

    
    # velocities #
    """# x
    Part_vel[0:Pop_A_num,0] = 1.0 - Vel_disp/2 + Vel_disp * np.random.rand(Pop_A_num)  # velocity around which it is centered (1.0, -1.0) could be added as a parameter in config
    Part_vel[Pop_A_num:,0] = -1.0 - Vel_disp/2 + Vel_disp * np.random.rand(Pop_B_num)
    
    # y
    Part_vel[0:Pop_A_num,1] = 1.0 - Vel_disp/2 + Vel_disp * np.random.rand(Pop_A_num)
    Part_vel[Pop_A_num:,1] = -1.0 - Vel_disp/2 + Vel_disp * np.random.rand(Pop_B_num)"""
    
    
    # rotation #
    x0, y0 = L/2, L/2        # Center of rotation
    
    x = Part_pos[:,0]
    y = Part_pos[:,1]
        
    x_c = np.angle(np.mean(np.exp(1j*2*np.pi*x/L))) * L/(2*np.pi)
    y_c = np.angle(np.mean(np.exp(1j*2*np.pi*y/L))) * L/(2*np.pi)
        
    center = [x_c, y_c]

    dx = Part_pos[:,0] - x_c  # x component of the r vector
    dy = Part_pos[:,1] - y_c  # y component of the r vector

    Part_vel[:,0] +=  omega * dy
    Part_vel[:,1] -=  omega * dx
    
    v0 = 20
    Vel_disp = 0.1
    
    Part_pos[:,0] = (x - x_c + L/2) % L
    Part_pos[:,1] = (y - y_c + L/2) % L

    """x0, y0 = L/2, L/2

    dx = (Part_pos[:,0] - x0 + L/2) % L - L/2
    dy = (Part_pos[:,1] - y0 + L/2) % L - L/2
    
    r = np.sqrt(dx**2 + dy**2) + 1e-10
    
    v_theta = v0 * r / (r + 0.05*L)
    
    # tangential velocity
    Part_vel[:,0] += -v_theta * dy / r
    Part_vel[:,1] +=  v_theta * dx / r"""
    
    Part_vel += (np.random.rand(Part_num,2) - 0.5) * Vel_disp

    Part_mass = np.ones(Part_num) * m

    Cell_pot = np.zeros(Cell_num)

    #one particle type
    rho0 = Part_num / Cell_num # Intrinsic charge density

    #test of initial conditions
    Part_per_cell, b = np.histogram(Part_pos, bins=Cell_edges)
    
    return Part_pos, Part_vel, Part_mass, Cell_pot, rho0, Part_per_cell