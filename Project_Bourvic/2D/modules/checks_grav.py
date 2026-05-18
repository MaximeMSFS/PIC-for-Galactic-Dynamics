"""
Set of functions for validation and extras
This is ongoing development and features may be broken
many function which I am not familiar with and still discovering are used, integration may be erroneous
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def energy(vel, rho, pot, parameters):
    
    
    # ========================================================================
    
    p, d = parameters["particles"], parameters["domain"]

    m = p["m"]
    dx = d["dx"]

    # ========================================================================
    
    
    E_kin = 0.5 * m * np.sum(vel**2)
    E_pot = 0.5 * np.sum(rho*pot) * dx**2
    E_tot = E_kin + E_pot
    

    return E_tot

def angular_momentum(pos, vel, parameters):
    
    
    # ========================================================================
    
    p, d = parameters["particles"], parameters["domain"]

    m = p["m"]
    L = d["L"]

    # ========================================================================
    
    x = pos[:,0]
    y = pos[:,1]
        
    x_c = np.angle(np.mean(np.exp(1j*2*np.pi*x/L))) * L/(2*np.pi)
    y_c = np.angle(np.mean(np.exp(1j*2*np.pi*y/L))) * L/(2*np.pi)
        
    center = [x_c, y_c]
    
    r = pos - center
    r = r - L * np.round(r/L)
    
    Lz = m * (r[:,0]*vel[:,1] - r[:,1]*vel[:,0])
    
    Lz_tot = np.sum(Lz)
    
    
    return Lz_tot
 

def Toomre(pos, vel, parameters, nb=600):

    
    # ========================================================================
    
    p, d = parameters["particles"], parameters["domain"]

    m = p["m"]
    L = d["L"]
    G = 1

    # ========================================================================
    
    
    b = np.linspace(0, L, num=nb+1)
    bcen = 0.5 * (b[1:] + b[:-1])
    X, Y = np.meshgrid(bcen, bcen, indexing='ij')
    dx = L / nb
    m = np.ones(len(pos[:,0])) * m
    
    x_pos = pos[:,0]
    y_pos = pos[:,1]
    
    x_c = np.angle(np.mean(np.exp(1j*2*np.pi*x_pos/L))) * L/(2*np.pi)
    y_c = np.angle(np.mean(np.exp(1j*2*np.pi*y_pos/L))) * L/(2*np.pi)
        
    center = [x_c, y_c]
    
    r = pos - center
    r = r - L * np.round(r/L)
    
    Xc = X - x_c
    Yc = Y - y_c
    
    Xc = Xc - L * np.round(Xc / L)
    Yc = Yc - L * np.round(Yc / L)
    
    Sigma, i, j, k = stats.binned_statistic_2d(x_pos, y_pos, m, statistic='sum', bins=b)
    Sigma = Sigma / (dx**2)
    
    mean_vx, i, j, k = stats.binned_statistic_2d(x_pos, y_pos, vel[:, 0], statistic='mean', bins=b)
    mean_vy, i, j, k = stats.binned_statistic_2d(x_pos, y_pos, vel[:, 1], statistic='mean', bins=b)
    
    var_vx, i, j, k = stats.binned_statistic_2d(x_pos, y_pos, vel[:, 0], statistic='std', bins=b)
    var_vy, i, j, k = stats.binned_statistic_2d(x_pos, y_pos, vel[:, 1], statistic='std', bins=b)
    
    var_vx = np.nan_to_num(var_vx)
    var_vy = np.nan_to_num(var_vy)
    var_vx = var_vx**2
    var_vy = var_vy**2
    disp = np.sqrt((var_vx + var_vy) / 2.0)
    
    mean_vx = np.nan_to_num(mean_vx)
    mean_vy = np.nan_to_num(mean_vy)
    
    v_theta = (Xc* mean_vy - Yc * mean_vx) / (np.sqrt(Xc**2 + Yc**2) + 1e10)
    #v_theta = (r[:,0] * mean_vy - r[:,1] * mean_vx) / (np.sqrt(r[:,0]**2 + r[:,1]**2) + 1e10)
    
    Omega = np.abs(v_theta) / (np.sqrt(Xc**2 + Yc**2) + 1e10)
    kappa = np.sqrt(2) * Omega
    
    Q = (kappa * disp) / (3.36 * G * Sigma + 1e-10)
    
    mask = Sigma < (np.max(Sigma) * 0.01)
    Q[mask] = np.nan
    
    
    return Q, X, Y

    
def ostriker_peebles(pos, vel, rho, pot, parameters):
    
    
    # ========================================================================
    
    p, d = parameters["particles"], parameters["domain"]

    m = p["m"]
    L, dx = d["L"], d["dx"]

    # ========================================================================
    
    x = pos[:,0]
    y = pos[:,1]
        
    x_c = np.angle(np.mean(np.exp(1j*2*np.pi*x/L))) * L/(2*np.pi)
    y_c = np.angle(np.mean(np.exp(1j*2*np.pi*y/L))) * L/(2*np.pi)
        
    center = [x_c, y_c]
    
    
    W = 0.5 * np.sum(rho*pot) * dx**2
    
    r = pos - center
    r = r - L * np.round(r/L)
    
    v_theta = (r[:,0] * vel[:,1] - r[:,1] * vel[:,0]) / (np.sqrt(r[:,0]**2 + r[:,1]**2) + dx/2)
    
    T = np.sum(0.5 * m * v_theta**2)
    
    criterion = np.abs(T / W)
   
    
    return criterion

def Jeans_mass():
    
    
    return 

def rotation_curve(pos, vel, parameters, nb=64):
    
    
    # ========================================================================
    
    p, d = parameters["particles"], parameters["domain"]

    m = p["m"]
    L, dx = d["L"], d["dx"]

    # ========================================================================
    
    
    x = pos[:,0]
    y = pos[:,1]
        
    x_c = np.angle(np.mean(np.exp(1j*2*np.pi*x/L))) * L/(2*np.pi)
    y_c = np.angle(np.mean(np.exp(1j*2*np.pi*y/L))) * L/(2*np.pi)
        
    center = [x_c, y_c]
    
    r = pos - center
    r = r - L * np.round(r/L)

    v_theta = (r[:,0] * vel[:,1] - r[:,1] * vel[:,0]) / (np.sqrt(r[:,0]**2 + r[:,1]**2) + 1e10)
    
    b = np.linspace(0, L/2, nb+1)
    bcen = 0.5 * (b[1:] + b[:-1])
    
    r = np.sqrt(r[:,0]**2 + r[:,1]**2)

    Part_per_bin, b = np.histogram(r, bins=b)
    v_rot, b = np.histogram(r, bins=b, weights=v_theta)
    v_rot2, b = np.histogram(r, bins=b, weights=v_theta**2)
    
    div = np.where(Part_per_bin > 0, Part_per_bin, 1)
    v_rot_mean = np.where( Part_per_bin > 0, v_rot / div, 0)
    v_rot2_mean = np.where( Part_per_bin > 0, v_rot2 / div, 0)

    sigma = np.sqrt(np.abs(v_rot2_mean - v_rot_mean**2)) #np.abs not rigorous, here to solve invalid value, to be patched later on


    return bcen, v_rot_mean, sigma