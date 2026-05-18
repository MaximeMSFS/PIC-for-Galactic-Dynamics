"""
plotting functions
"""
import matplotlib.pyplot as plt

def conditions_plot(Cell_pos, Part_per_cell, Part_pos, Part_vel, instant, time, parameters):   
    
    # ========================================================================
    
    Pop_A_num = parameters["particles"]["Pop_A_num"]
    Pop_B_num = parameters["particles"]["Pop_B_num"]
    
    # ========================================================================   
    
    plt.figure()
    plt.suptitle(f"{instant} Conditions\n t = {time} s", fontweight="extra bold")

    plt.subplot(1,2,1)
    plt.scatter(Cell_pos, Part_per_cell, marker='o', s=0.5, color='k' )
    plt.xlabel("Position")
    plt.ylabel("Particles per cell")
    plt.title("Particle distribution", fontweight='bold')
    plt.grid()

    plt.subplot(1,2,2)
    plt.scatter(Part_pos[0:Pop_A_num], Part_vel[0:Pop_A_num], label='population A', s=0.1, color='blue')
    plt.scatter(Part_pos[Pop_A_num:], Part_vel[Pop_A_num:], label='population B', s=0.1, color='red')
    plt.xlabel("Position")
    plt.ylabel("velocity")
    plt.legend()
    plt.title("Phase Space Distribution", fontweight='bold')
    plt.grid()

    plt.tight_layout()
    plt.show()
       
    return




def field_plot(Cell_pos, V, Cell_field, instant, time, parameters):  
    
    plt.figure()
    plt.suptitle(f"{instant} Electrical Potential and Field\n t = {time} s", fontweight="extra bold")
    plt.subplot(1,2,1)
    plt.plot(Cell_pos, V)
    plt.xlabel("Position")
    plt.ylabel("potential")
    plt.title("Potential per Cell" , fontweight='bold')
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(Cell_pos, Cell_field)
    plt.xlabel("Position")
    plt.ylabel("field")
    plt.title("Field per Cell", fontweight='bold')
    plt.grid()

    plt.tight_layout()
      
    return




def energy_plot(energy, time, parameters):    
    
    plt.figure()
    plt.plot(time, energy)
    plt.xlabel("Time")
    plt.ylabel("Total Energy")
    plt.title("Energy deviation", fontweight='bold')
    plt.grid()
        
    return




def position_plot(pos, parameters):
    
    # ========================================================================

    Pop_A_num = parameters["particles"]["Pop_A_num"]
    
    # ========================================================================
       
    plt.figure()
    plt.scatter(pos[0:Pop_A_num,0], pos[0:Pop_A_num,1], color='blue')
    plt.scatter(pos[Pop_A_num:,0], pos[Pop_A_num:,1], color='red')
    plt.xlabel("x coordinate (m)")
    plt.ylabel("x coordinate (m)")
    plt.axis('equal')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("Particles Position", fontweight='bold')
    plt.grid()
      
    return