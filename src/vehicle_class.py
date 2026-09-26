import numpy as np
from matplotlib import pyplot as plt
from IPython import embed as IPS
import matplotlib.animation as animation
import matplotlib.image as mpimg
from tqdm import tqdm
import torch

class P_controller:
    def __init__(self, K_p, K_i, d_ref, dt):
        self.K_p = K_p
        self.K_i = K_i
        self.d_ref = d_ref
        self.dt = dt

    def compute_error(self,  my_distance_to_front, fronts_distance_to_front, backs_distance_to_front ,errors_vehicle):
        if backs_distance_to_front is not None:  # not the last vehicle
            # Based on Laplace matrix
            self.error = 2*my_distance_to_front - backs_distance_to_front  + fronts_distance_to_front  # *(1-int(fronts_distance_to_front==self.d_ref))
            self.error_sum = sum(errors_vehicle)
        else:  # last vehicle
            self.error = my_distance_to_front  + fronts_distance_to_front
            self.error_sum = sum(errors_vehicle)
    def return_torque(self, my_distance_to_front, fronts_distance_to_front, backs_distance_to_front, errors_vehicle):
        self.compute_error(my_distance_to_front, fronts_distance_to_front, backs_distance_to_front, errors_vehicle)
        self.torque = self.K_p*self.error + self.K_i*self.error_sum*self.dt
        return self.torque
    def return_error(self):
        return self.error

class vehicle:
    '''
    Initialize a vehicle with the following parameters:
    r ... wheel radius
    ga ... gravitational acceleration
    alpha ... road grade
    cr ... rolling resistance coefficient
    rho ... air density
    Ar ... cross-sectional area of the vehicle
    Cd ... aerodynamic drag coefficient
    m ... mass
    dt ... discrete time step
    dummy_param ... some dummy parameter that does not influence the model
    Of these parameters we assume that we have knowledge of m, alpha, and some dummy parameter
    '''
    def __init__(self,r,ga,alpha,cr,rho,Ar,Cd,m,dt, s_init):
        self.r = r
        self.ga = ga
        self.alpha = alpha
        self.cr = cr
        self.rho = rho
        self.Ar = Ar
        self.Cd = Cd
        self.m = m
        self.dt = dt
        self.v = 0
        self.s = s_init
        self.x = np.vstack((self.s, self.v))
        self.max_speed = np.inf
        self.max_torque = 50000
        self.u = 0

    def dynamics(self,u):
        # u = np.clip(u, -self.max_torque, self.max_torque)  # Limit torque within valid range; anyways input
        self.s = self.s + self.v*self.dt
        # self.v = min(self.v + ((self.Ft(u) + self.Fg() + self.Fr() + self.Fd())/self.m)*self.dt + np.random.normal(0,1e-4,1), self.max_speed)  # Limit velocity within limit 
        self.v = self.v + ((self.Ft(u) + self.Fg() + self.Fr() + self.Fd())/self.m)*self.dt
        self.x = np.vstack((self.s,self.v))

    # Gravitational Force
    def Fg(self):
        return -self.m*self.ga*np.sin(self.alpha)

    # Rolling Resistance
    def Fr(self):
        if self.v > 1e-2:
            return -self.cr*self.m*self.ga 
        elif self.v < -1e-2:
            return self.cr*self.m*self.ga 
        else:
            return 0

    # Aerodynamic Drag
    def Fd(self):
        return -0.5*self.rho*self.Ar*self.Cd*self.v**2

    # Forward Torque
    def Ft(self,u):
        return (1/self.r)*u

class leading_vehicle(vehicle):
    def __init__(self,r,ga,alpha,cr,rho,Ar,Cd,m, dt, s_init, v):
        super().__init__(r,ga,alpha,cr,rho,Ar,Cd,m,dt,s_init)
        self.v = v
    def dynamics(self):
        # u = np.clip(u, -self.max_torque, self.max_torque)  # Limit torque within valid range; anyways input
        self.s = self.s + self.v*self.dt
        self.x = np.vstack((self.s,self.v))


def simulate(hyperparameters, K_p_values, K_i_values):
    num_vehicles, v_leader, d_ref, steps, dt, s_init_list = hyperparameters
    list_vehicles = []
    list_controllers = []
    for i in range(num_vehicles):
        r = np.random.uniform(0.4, 0.6)   # 0.5   # Wheel radius (m)
        ga = 9.81      # Gravitational acceleration (m/s²); all on same planet
        alpha = 0  # 0.05   # Road grade (rad) ≈ 2.86°; all on same street
        cr = np.random.uniform(0.004, 0.008)  # 0.006     # Rolling resistance coefficient
        rho = 1.225    # Air density (kg/m³); all in the same air
        Ar = np.random.uniform(5, 7) # 10        # Cross-sectional area (m²)
        Cd = np.random.uniform(0.4, 0.8)  # 0.6       # Aerodynamic drag coefficient
        m = np.random.uniform(1950, 2050)  # 2000      # Mass (kg) ~15 tons
        if i != num_vehicles - 1:  # not the leading vehicle with const. velocity dynamics
            list_vehicles.append(vehicle(r, ga, alpha, cr, rho, Ar, Cd, m, dt, s_init=s_init_list[i]))
            list_controllers.append(P_controller(K_p=K_p_values[i], K_i=K_i_values[i], d_ref=d_ref, dt=dt))
        else:
            list_vehicles.append(leading_vehicle(r, ga, alpha, cr, rho, Ar, Cd, m, dt, s_init=s_init_list[i], v=v_leader))  # sinit 250

    # Simulation settings
     # Number of simulation steps
    
    initial_torque = 0

    # Set up controller for each agent
    positions = np.zeros([steps, num_vehicles])
    distances_to_front_vehicle = np.zeros([steps, num_vehicles])
    distances_to_front_vehicle[:, -1] = d_ref  # first leading vehicle; but we count from left to right
    velocities = np.zeros([steps, num_vehicles])
    total_error_list = [0]*steps
    torques = np.zeros([steps, num_vehicles-1])
    abs_errors = np.zeros([steps, num_vehicles])

    torque = initial_torque
    for i in range(steps):
        for j in range(num_vehicles):  # update position and velocity for all vehicles.
            positions[i, j] = list_vehicles[j].s
            velocities[i, j] = list_vehicles[j].v
        for j in range(num_vehicles-1):  # now compute all distances!; we already know the last entry
            distances_to_front_vehicle[i, j] = positions[i, j+1] - positions[i, j]
        for j in range(num_vehicles):  # now compute inputs etc.
            if j != num_vehicles - 1:  # not the leading vehicle
                my_distance_to_front = distances_to_front_vehicle[i, j]
                fronts_distance_to_front = distances_to_front_vehicle[i, j+1]
                if j != 0:  # not the first vehicle
                    backs_distance_to_front = distances_to_front_vehicle[i, j-1]
                    torque = list_controllers[j].return_torque(my_distance_to_front, fronts_distance_to_front,
                     backs_distance_to_front, errors_vehicle=abs_errors[:, j])
                if j == 0:
                    torque = list_controllers[j].return_torque(my_distance_to_front, fronts_distance_to_front,
                     backs_distance_to_front=None, errors_vehicle=abs_errors[:, j])
                list_vehicles[j].dynamics(torque)  # Apply torque
                abs_errors[i, j]=abs(list_controllers[j].error)
                total_error_list[i] += list_controllers[j].error
                torques[i, j]=torque
            else:  # if leading vehicle
                list_vehicles[j].dynamics()
    return positions, velocities, total_error_list, torques, distances_to_front_vehicle, abs_errors

def reward_function(distances_to_front_vehicle, d_ref, num_vehicles, steps, T):
    deviation = np.abs(distances_to_front_vehicle - d_ref)
    abs_error = np.sum(deviation)
    avg_error = 1/(num_vehicles-1)*1/steps*abs_error
    min_distance = np.min(np.abs(distances_to_front_vehicle))
    collision_penalty = (num_vehicles-1)*(min_distance-d_ref)
    reward = - avg_error - T*collision_penalty
    return reward

class platoon_simulator:
    def __init__(self, num_vehicles, v_leader, d_ref, steps, dt, s_init_list, T):
        self.num_vehicles = num_vehicles
        self.d_ref = d_ref
        self.steps = steps
        self.T = T
        self.hyperparameters = [num_vehicles, v_leader, d_ref, steps, dt, s_init_list]
        
        # Dummy variables to keep the BO plotting 
        self.fX = torch.zeros(1) 
        self.X_center = torch.zeros(1, 1)
    
    def conduct_experiment(self, x):
        # Scale the BO's [0, 1] output to a realistic K_p range [0.0, 2.0]
        K_p_values = (x.flatten().numpy() * 2.0).tolist() 
        K_i_values = [0.001] * self.num_vehicles
        
        # Run your simulation
        positions, velocities, total_error_list, torques, distances_to_front_vehicle, abs_errors = simulate(
            self.hyperparameters, K_p_values, K_i_values
        )
        
        # Calculate Reward using your exact function
        reward = reward_function(distances_to_front_vehicle, self.d_ref, self.num_vehicles, self.steps, self.T)
        
        # Calculate Constraint (Minimum physical distance to avoid crash)
        follower_distances = distances_to_front_vehicle[:, :-1]
        min_distance = np.min(follower_distances)
        
        return reward, min_distance

    def f(self, x):
        # The Objective GP calls this to get the Reward
        reward, _ = self.conduct_experiment(x)
        return reward

    def c(self, x):
        # The Constraint GP calls this to get the safety distance
        _, min_distance = self.conduct_experiment(x)
        return min_distance
    
if __name__ == '__main__':
    s_init_list = [0, 25, 38, 59, 100]
    num_vehicles = 5
    v_leader = 10
    T = 500 # Total simulation time in seconds
    dt = 0.2
    steps = int(T / dt) 
    # Determine goal distance
    d_ref = 10 # we want 10m between the LKWs
    time = np.linspace(0, T, steps)
    K_p_values = [0.5]*num_vehicles  # have this between 0 and 2!
    K_i_values = [0.00]*num_vehicles
    hyperparameters_simulation = [num_vehicles, v_leader, d_ref, steps, dt, s_init_list]
    positions, velocities, total_error_list, torques, distances_to_front_vehicle, abs_errors  = simulate(hyperparameters_simulation, K_p_values, K_i_values)  # we will tune K_p values from our algoritm
    deviation = np.abs(distances_to_front_vehicle - d_ref)
    abs_error = np.sum(deviation)
    avg_error = 1/(num_vehicles-1)*1/steps*abs_error
    min_distance = np.min(np.abs(distances_to_front_vehicle))
    safety_threshold = -d_ref*(num_vehicles-1)*T  # I guess quite non-smooth
    reward = reward_function(distances_to_front_vehicle, d_ref, num_vehicles, steps, T)
    

    # Plot results
    plt.figure()
    for j in range(num_vehicles):
        plt.plot(time, positions[:, j], label=f"LKW {j}")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time, total_error_list)
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.grid()
    plt.show()


    plt.figure()
    for j in range(num_vehicles):
        plt.plot(time, distances_to_front_vehicle[:, j], label=f'{j}')
    plt.xlabel("Time (s)")
    plt.legend()
    plt.ylabel("Distances")
    plt.grid()
    plt.show()

    plt.figure()
    for j in range(num_vehicles):
        plt.plot(time, velocities[:, j], label=f'{j}')
    plt.xlabel("Time (s)")
    plt.legend()
    plt.ylabel("Velocity (m/s)")
    plt.grid()
    plt.show()

    plt.figure()
    for j in range(num_vehicles-1):
        plt.plot(time, torques[:, j], label=f'{j}')
    plt.xlabel("Time")
    plt.legend()
    plt.ylabel("Torque")

    plt.figure()
    for j in range(num_vehicles-1):
        plt.plot(time, abs_errors[:, j], label=f'{j}')
    plt.xlabel("Time")
    plt.legend()
    plt.ylabel("Error")



    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["b", "r", "g", "m", "k"]  # Blue, Red, Green, Magenta
    ax.set_xlim(np.min(positions)-10, np.max(positions) + 10)  # Extend the x-axis dynamically
    ax.set_ylim(-0.2, 0.2)  # Keep the y-axis fixed
    ax.set_xlabel("Position (m)")
    ax.set_title("LKW Movement Simulation")

    # Draw the road
    road, = ax.plot([np.min(positions)-10, np.max(positions) + 10], [0, 0], "k-", linewidth=2)

    vehicle_img = mpimg.imread("truck.png")  # PNG with transparent background recommended
    # x = 200  # Arbitrary position along the x-axis
    # # ax.set_ylim(-10, 10)
    # test_marker = ax.imshow(vehicle_img, extent=(x - 50, x + 50, -50, 50), zorder=10)
    # plt.show()
    # Initialize markers for all LKWs with different colors
    LKW_markers = [
        ax.plot([], [], f"{colors[j]}o", markersize=10, label=f"LKW {j+1}")[0] 
        for j in range(num_vehicles)
    ]
    # LKW_markers = [
    #     ax.imshow(vehicle_img, extent=(0, 2, -0.5, 0.5), zorder=10)  # Initial dummy extent
    #     for _ in range(num_vehicles)
    # ]


    # Update function for animation
    def update(frame):
        for j in range(num_vehicles):
            LKW_markers[j].set_data([positions[frame, j]], [0])  # Move each LKW along the x-axis
            # x = positions[frame, j]
            # LKW_markers[j].set_extent((x - 100, x + 100, -3, 3))
        return LKW_markers

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=50, blit=True)
    # ani.save("lkw_simulation.gif", writer="pillow", fps=20)
    # plt.legend()
    plt.show()
    print("Done")







    # plt.legend()
    # plt.show()



    