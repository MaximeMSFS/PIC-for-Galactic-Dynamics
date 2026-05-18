"""
routine for the user input
"""

from pathlib import Path
from colorama import init, Fore, Style

import modules.file_handling as fh
from modules.config import config

init(autoreset=True)

def User_input():
    Invalid_entry = True
    while Invalid_entry :
        init = input(Fore.CYAN + Style.BRIGHT + "Do you wish to resume calculation from a previously computed file ?\n (Yes, No) :\n").strip().lower()
        if (init == 'yes') or (init == 'no') :
            Invalid_entry = False
            if (init == "yes") :
                input_dir = input(Fore.CYAN + "Please enter the directory of the saved files :\n").strip()
                parameters, step, time, pos, vel, pot = fh.read_config(input_dir)
                print (Fore.YELLOW + Style.BRIGHT + "/!\ Partially Working /!\ \n")
                print (Fore.WHITE + Style.BRIGHT + f"\nResuming simulation from step {step} at t = {time} s\n")
            else :
                parameters = config()
        else :
            print (Fore.RED + 'Invalid entry')
            
    if (init == 'no') : 
        Invalid_entry = True
        while Invalid_entry :
            scheme = input(Fore.CYAN + Style.BRIGHT + "\nWhich scheme do you wish to use ?\n (0 = Explicite Euler, 1 = Implicit Euler, 2 = Leapfrog, 3 = RK4) :\n").strip()
            if (scheme == '0') or (scheme == '1') or (scheme == '2') or (scheme == '3') :
                Invalid_entry = False
                scheme = int(scheme)
                parameters['numerical']['scheme'] = scheme
            else :
                print (Fore.RED + Style.BRIGHT + 'Invalid entry')
        
        Invalid_entry = True
        while Invalid_entry :
            solver = input(Fore.CYAN + Style.BRIGHT + "\nWhich solver do you wish to use ?\n (F = Fourrier, P = Poisson) :\n").strip().lower()
            if (solver == 'f') or (solver == "p") :
                Invalid_entry = False
                parameters['numerical']['solver'] = solver
            else :
                print (Fore.RED + 'Invalid entry')
                
        Invalid_entry = True
        while Invalid_entry :
            variable_dt = input(Fore.CYAN + Style.BRIGHT + "\nDo you wish to use a variable time step ?\n (Yes, No) :\n").strip().lower()
            if (variable_dt == 'yes') or (variable_dt == 'no') :
                Invalid_entry = False
                parameters['numerical']['variable_dt'] = variable_dt
            else :
                print (Fore.RED + 'Invalid entry')
                
    else :
        scheme = parameters['numerical']['scheme']
        solver = parameters['numerical']['solver']
        variable_dt = parameters['numerical']['variable_dt']
    
    
            
    
    Invalid_entry = True
    while Invalid_entry :
        save = input(Fore.CYAN + Style.BRIGHT + "\nDo you wish to save the results ?\n (Yes, No) :\n").strip().lower()
        if (save == 'yes') or (save == "no") :
            Invalid_entry = False
            if (save == "yes") :
                output_dir = input(Fore.CYAN + Style.BRIGHT + "Please enter the output directory name :\n").strip()
                data_dir = Path("data")/output_dir
                data_dir.mkdir(parents=True, exist_ok=True)
                main_file = open(data_dir/f"{output_dir}_main.dat", 'w')
                pos_file = open(data_dir /f"{output_dir}_position.dat", "ab")
                vel_file = open(data_dir /f"{output_dir}_velocity.dat", "ab")
                pot_file = open(data_dir /f"{output_dir}_potential.dat", "ab")
                resume_file = data_dir /f"{output_dir}_resume.npz"
                fh.header(main_file, parameters)
            else :
                main_file, pos_file, vel_file, pot_file, resume_file = "", "", "", "", ""
        else :
            print (Fore.RED + 'Invalid entry')
    
    if (scheme == 0) :
        print (Fore.WHITE + Style.BRIGHT + '\nUsing Explicite Euler Scheme ', end="")
    elif  (scheme == 1) :
        print (Fore.WHITE + Style.BRIGHT + '\nUsing Implicit Euler Scheme ', end="")
    elif  (scheme == 2) :
        print (Fore.WHITE + Style.BRIGHT + '\nUsing Leapfrog Scheme ', end="")
    else :
        print (Fore.WHITE + Style.BRIGHT + '\nUsing Range-Kutta 4 Scheme ', end="")
    
        
    if (solver == 'f') :
        print (Fore.WHITE + Style.BRIGHT + 'with the Fourrier Solver ', end="")
    else :
        print (Fore.WHITE + Style.BRIGHT + 'with the Poisson Solver ', end='')
        
    if (variable_dt == 'yes') :
        print (Fore.WHITE + Style.BRIGHT + 'and variable time step', end='')
    
    if (save == "yes") :
        print (Fore.WHITE + Style.BRIGHT + f'\nSaving results to {data_dir}')
    print ('\n')
    
    return parameters, main_file, pos_file, vel_file, pot_file, resume_file, save