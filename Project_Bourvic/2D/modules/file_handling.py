"""
file handling functions for saving and starting back from an old computation
"""
import numpy as np
from pathlib import Path

def header(file, parameters):
    
    
    # ========================================================================
    
    p, d, t, n = parameters["particles"], parameters["domain"], parameters["time"], parameters["numerical"]

    # particles #
    Part_num, Pop_A_num, Pop_B_num = p["Part_num"], p["Pop_A_num"], p["Pop_B_num"]
    q, m = p["q"], p["m"]
    pos_disp, Vel_disp = p["pos_disp"], p["Vel_disp"]

    # domain #
    L, Cell_num = d["L"], d["Cell_num"]
    eps0 = d["eps0"]

    # time #
    Tmin, Tmax, dt0 = t["Tmin"], t["Tmax"], t["dt0"]

    # numerical #
    Relax_step_max, epsilon, eps = n["Relax_step_max"], n["epsilon"], n["eps"]
    save_step, random_seed = n["save_step"], n["random_seed"]
    scheme, solver = n["scheme"], n["solver"]
    
    # ========================================================================
    
    
    file.write("==============================================================\n")
    file.write("                  1D PLASMA SIMULATION OUTPUT                 \n")
    file.write("==============================================================\n")
    
    file.write("computed with Project_Main_7.py\n\n")
    
    file.write("--- Simulation Parameters ---\n")
    file.write(f"scheme              : {scheme}\n")
    file.write(f"solver              : {solver}\n\n")
    
    file.write("--------- Particles ---------\n")
    file.write(f"particles number    : {Part_num}\n")
    file.write(f"population A number : {Pop_A_num}\n")
    file.write(f"population B number : {Pop_B_num}\n")
    file.write(f"charge              : {q}\n")
    file.write(f"mass                : {m}\n")
    file.write(f"position dispersion : {pos_disp}\n")
    file.write(f"velocity dispersion : {Vel_disp}\n\n")
    
    file.write("----- Domain Parameters -----\n")
    file.write(f"domain length       : {L}\n")
    file.write(f"cell number         : {Cell_num}\n")
    file.write(f"Permitivity         : {eps0}\n\n")
    
    file.write("------ Time Parameters ------\n")
    file.write(f"Tmin                : {Tmin}\n")
    file.write(f"Tmax                : {Tmax}\n")
    file.write(f"dt                  : {dt0}\n\n")
 
    file.write("---- Numerical Parameters ----\n")
    file.write(f"max relaxation steps : {Relax_step_max}\n")
    file.write(f"epsilon              : {epsilon}\n")
    file.write(f"eps                  : {eps:.4f}\n")
    file.write(f"save every X step    : {save_step}\n")
    file.write(f"variable time step   : {save_step}\n")
    file.write(f"random seed          : {random_seed}\n\n")
    
    file.write("==============================================================\n\n")
    
    file.write(f"{'step':>6} {'time':>10} {'energy':>10}\n")
    
    
    return




def read_config(input_dir):
    
    
    data_dir = Path("data")/input_dir
    
    config = {"particles":{}, "domain":{},"time":{}, "fft":{}, "numerical":{}}

    with open(data_dir/f"{input_dir}_main.dat", "r") as f:
        lines = f.readlines()

    for line in lines:
        if ":" not in line:
            continue

        key, value = line.split(":")
        key = key.strip()
        value = value.strip()

        try:
            if "." in value or "e" in value.lower():
                value = float(value)
            else:
                value = int(value)
        except:
            pass

        # particles
        if key == "particles number":
            config["particles"]["Part_num"] = value
        elif key == "population A number":
            config["particles"]["Pop_A_num"] = value
        elif key == "population B number":
            config["particles"]["Pop_B_num"] = value
        elif key == "charge":
            config["particles"]["q"] = value
        elif key == "mass":
            config["particles"]["m"] = value
        elif key == "position dispersion":
            config["particles"]["pos_disp"] = value
        elif key == "velocity dispersion":
            config["particles"]["Vel_disp"] = value

        # domain
        elif key == "domain length":
            config["domain"]["L"] = value
        elif key == "cell number":
            config["domain"]["Cell_num"] = value
        elif key == "Permitivity":
            config["domain"]["eps0"] = value

        # time
        elif key == "Tmin":
            config["time"]["Tmin"] = value
        elif key == "Tmax":
            config["time"]["Tmax"] = value
        elif key == "dt":
            config["time"]["dt0"] = value
        elif key == "step":
            config["time"]["step"] = value

        # numerical
        elif key == "max relaxation steps":
            config["numerical"]["Relax_step_max"] = value
        elif key == "epsilon":
            config["numerical"]["epsilon"] = value
        elif key == "eps":
            config["numerical"]["eps"] = value
        elif key == "save every X step":
            config["numerical"]["save_step"] = value
        elif key == "random seed":
            config["numerical"]["random_seed"] = value
        elif key == "scheme":
            config["numerical"]["scheme"] = value
        elif key == "solver":
            config["numerical"]["solver"] = value
            
            
#============================ NEEDS TO BE CLEANED =============================            
            
    dx = config["domain"]["L"] / config["domain"]["Cell_num"]
    
    Cell_pos = np.linspace(0, config["domain"]["L"]-dx, config["domain"]["Cell_num"]) + dx/2 # position of each cell (center of the cell)
    Cell_edges = np.linspace(0, config["domain"]["L"], config["domain"]["Cell_num"]+1)       # position of each cell borders (used for bins)
    
    config["domain"]["Cell_pos"] = Cell_pos
    config["domain"]["Cell_edges"] = Cell_edges
    
    t = np.arange(config["time"]["Tmin"],config["time"]["Tmax"], config["time"]["dt0"])  # Times array
    
    config["time"]["t"] = t
    
    k = 2*np.pi*np.fft.fftfreq(config["domain"]["Cell_num"], d=dx)
    k2 = k**2
    k2[0] = 1.0   # avoids division by zero
    
    config["fft"] = dict(k = k, 
                         k2 = k2)
    data = np.load(data_dir/f"{input_dir}_resume.npz")

    step = int(data["step"])
    config["time"]["step"] = step
    
    resume_time = float(data["time"])
    pos = data["pos"]
    vel = data["vel"]
    pot = data["pot"]
#==============================================================================


    return config, step, resume_time, pos, vel, pot