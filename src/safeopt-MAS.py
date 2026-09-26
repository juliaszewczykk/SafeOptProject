
import sys
import os
import torch
import warnings
# warnings.filterwarnings("ignore")

import numpy as np
import gpytorch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import dill

random_seed_number = 107

np.random.seed(random_seed_number)

# Fix seed for PyTorch (CPU)
torch.manual_seed(random_seed_number)


# Add the relative path to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Print to verify
print("Changed working directory to:", os.getcwd())


from safebo_MAS_plot import plot_2D_mean, plot_reward, plot_3D_sampled_space, plot_1D_sampled_space, plot_2D_UCB
from pacsbo.pacsbo_main import compute_X_plot, ground_truth, initial_safe_samples, PACSBO, GPRegressionModel
from vehicle_class import platoon_simulator


def build_initial_positions(num_agents, spacing=15.0): # Added for initial position of the agents
    """Generate enough initial positions for any agent count."""
    return [spacing * i for i in range(num_agents)]


def acquisition_function(noise_std, delta_confidence, exploration_threshold, B, X_plot, X_sample, Y_sample, C_sample, t,
                         lengthscale_agent_spatio, a_parameter, lengthscale_temporal_RBF, lengthscale_temporal_Ma12,
                           output_variance_RBF, output_variance_Ma12, safety_threshold, constraint_threshold):
    
    def update_model(cube):
        cube.compute_model(gpr=GPRegressionModel)  # compute confidence intervals?
        cube.compute_mean_var()
        cube.compute_confidence_intervals_evaluation(RKHS_norm_guessed=B)

# We build two different GP - one for objective and one for safety constraint
    cube_objective = PACSBO(delta_confidence=delta_confidence, noise_std=noise_std, tuple_ik=(-1, -1), X_plot=X_plot, X_sample=X_sample,
                    Y_sample=Y_sample, iteration=t, safety_threshold=safety_threshold, exploration_threshold=exploration_threshold, compute_all_sets=False, lengthscale_agent_spatio=lengthscale_agent_spatio,
                    a_parameter=a_parameter, lengthscale_temporal_RBF=lengthscale_temporal_RBF, lengthscale_temporal_Ma12=lengthscale_temporal_Ma12,
                    output_variance_RBF=output_variance_RBF, output_variance_Ma12=output_variance_Ma12)  # all samples that we currently have
    
    cube_constraint = PACSBO(delta_confidence=delta_confidence, noise_std=noise_std, tuple_ik=(-1, -1), X_plot=X_plot, X_sample=X_sample,
                    Y_sample=C_sample, iteration=t, safety_threshold=constraint_threshold, exploration_threshold=exploration_threshold, compute_all_sets=False, lengthscale_agent_spatio=lengthscale_agent_spatio,
                    a_parameter=a_parameter, lengthscale_temporal_RBF=lengthscale_temporal_RBF, lengthscale_temporal_Ma12=lengthscale_temporal_Ma12,
                    output_variance_RBF=output_variance_RBF, output_variance_Ma12=output_variance_Ma12)  # all samples that we currently have
    
# We update both models
    update_model(cube_objective)
    update_model(cube_constraint)

#New logic to find safe set, maximizers and expanders combining the two
# Calculate safe set strictly from Constraint GP
    cube_constraint.compute_safe_set()
    
# Force the Objective GP to only explore what the Constraint GP deems safe
    cube_objective.S = cube_constraint.S.clone()
    
    cube_objective.M = cube_objective.S.clone()
    cube_objective.G = cube_objective.S.clone()
    # Find Expanders and Maximizers safely
    cube_objective.maximizer_routine(best_lower_bound_others=-np.inf)
    cube_objective.expander_routine()
    
# Choose next point based on what the objective wants to do  
    
    if cube_objective.safety_threshold > -np.inf:
        if sum(torch.logical_or(cube_objective.M, cube_objective.G)) != 0:
                max_indices = torch.nonzero(cube_objective.ucb[torch.logical_or(cube_objective.M, cube_objective.G)] == torch.max(cube_objective.ucb[torch.logical_or(cube_objective.M, cube_objective.G)]), as_tuple=True)[0]
                random_max_index = max_indices[torch.randint(len(max_indices), (1,))].item()
                x_new = cube_objective.discr_domain[torch.logical_or(cube_objective.M, cube_objective.G)][random_max_index, :]
        else:
            # warnings.warn('No new input found. Returning last point of X_sample')
            x_new = torch.cat((X_sample[-1], cube_objective.iteration.flatten() + 1))  # .unsqueeze(0)
    else:
        if sum(cube_objective.M) != 0:  
            max_indices = torch.nonzero(cube_objective.ucb[cube_objective.M] == torch.max(cube_objective.ucb[cube_objective.M]), as_tuple=True)[0]
            random_max_index = max_indices[torch.randint(len(max_indices), (1,))].item()
            x_new = cube_objective.discr_domain[cube_objective.M][random_max_index, :]
        else:
            # warnings.warn('No new input found. Returning last point of X_sample')
            x_new = torch.cat((X_sample[-1], cube_objective.iteration.flatten() + 1))

    return x_new, cube_objective

# We add C_sample in the hyperparameters so Y_sample stores the objective score and C_sample stores the constraint scores
def process_agent(j, X_plot, communication_indices_list, X_sample_full, Y_sample, C_sample, t, hyperparameters):
    noise_std, delta_confidence, exploration_threshold, RKHS_norm, lengthscale_agent_spatio, a_parameter, lengthscale_temporal_RBF, lengthscale_temporal_Ma12, output_variance_RBF, output_variance_Ma12, safety_threshold, constraint_threshold = hyperparameters
    X_sample = X_sample_full[:, communication_indices_list]

    x_new_neighbors, cube = acquisition_function(
        noise_std, delta_confidence, exploration_threshold, RKHS_norm,
        X_plot, X_sample, Y_sample, C_sample, t, lengthscale_agent_spatio,
        a_parameter, lengthscale_temporal_RBF, lengthscale_temporal_Ma12,
        output_variance_RBF, output_variance_Ma12, safety_threshold, constraint_threshold
    )
    cube_dict = {}
    cube_dict['iteration'] = cube.iteration
    cube_dict['mean'] = cube.mean
    # cube_dict['discr_domain'] = cube.discr_domain
    cube_dict['x_sample'] = cube.x_sample
    cube_dict['var'] = cube.var  # maybe also not necessary, let's see
    cube_dict['y_sample'] = cube.y_sample
    cube_dict['c_sample'] = C_sample  
    cube_dict['constraint_threshold'] = constraint_threshold
    cube_dict['safety_threshold'] = safety_threshold
    cube_dict['beta'] = cube.beta
    if communication:
        if full_communication:
            x_new = x_new_neighbors
        else:
            x_new = x_new_neighbors[1].unsqueeze(0) if j != 0 else x_new_neighbors[0].unsqueeze(0)  # I think this is fine for any communication

    else: 
        x_new = x_new_neighbors[0].unsqueeze(0)
    agents_j = [
        x_new.detach(),  # Detach tensor
        x_new_neighbors.detach(),  # Detach tensor
        cube_dict
    ]
    return j, agents_j


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)  # Ensure safe multiprocessing with PyTorch
    # Generate ground truth
    iterations = 50
    num_agents = 8
    random_expert = False
    sequential_expert = False
    agents = {}
    communication_list_dict = {}
    X_plot_dict = {}
    noise_std = 1e-2  # increase a little for numerical stability
    delta_confidence = 0.9
    exploration_threshold = 0.1
    dimension = num_agents
    initial_point_quantile = 0.5
    safety_quantile = 0.2
    communication = True
    full_communication = True
    time_latent_variable = False

    '''
    Hyperparameters
    '''
    RKHS_norm_spatio_temporal = 1  # 0.1
    lengthscale_agent_spatio = 0.3  # 0.5
    a_parameter = 50  # weighting factor for the brownian motion and reverse brownian motion kernel
    lengthscale_temporal_RBF = 20  # 20  # 5
    lengthscale_temporal_Ma12 = 5  # 5  # 1
    # Changing length scales did not directly influence stuff
    output_variance_RBF = 0.1  # 0.1 0.5  # 0.5
    output_variance_Ma12 = 0.1  # 0.1 2  # num_agents
    # Changing the output variance does influence stuff a lot
    #constraint_threshold = 1.0  # NEW: Minimum safe bumper distance
    #lengthscale_gt = num_agents/10
    #gt = ground_truth(num_center_points=1000, dimension=dimension, RKHS_norm=1, lengthscale=lengthscale_gt)    
    #safety_threshold = torch.quantile(gt.fX, safety_quantile).item()  # -np.inf  # 
    v_leader = 10
    T_sim = 50  # 50 seconds to keep optimization fast
    dt = 0.1
    steps = int(T_sim / dt) 
    d_ref = 10 
    #    s_init_list = [0, 15, 30, 45, 60, 75, 90, 105] # 8 starting positions for 8 agents
    s_init_list = build_initial_positions(num_agents, spacing=15.0)
    
    gt = platoon_simulator(num_agents, v_leader, d_ref, steps, dt, s_init_list, T_sim)
    
    safety_threshold = -np.inf  # We don't bound the performance score
    constraint_threshold = 1.0
    print(f'The heuristic maximum of the function is {max(gt.fX)} and located at {gt.X_center[torch.argmax(gt.fX)]}.')
    print(f'The safety threshold is {safety_threshold}.')

    hyperparameters = [noise_std, delta_confidence, exploration_threshold, RKHS_norm_spatio_temporal, lengthscale_agent_spatio,
                    a_parameter, lengthscale_temporal_RBF, lengthscale_temporal_Ma12, output_variance_RBF, 
                    output_variance_Ma12, safety_threshold, constraint_threshold]
    agents['gt'] = gt
    # Finding initial safe sample
    while True:
        X_sample_full = torch.rand(dimension).unsqueeze(0)  # just start here
        # X_sample_full = gt.X_center[torch.argmax(gt.fX)].unsqueeze(0)  # start with highest point
        Y_sample = torch.tensor([float(gt.f(X_sample_full))], dtype=torch.float32)
        C_sample = torch.tensor([float(gt.c(X_sample_full))], dtype=torch.float32) #NEW
        if Y_sample.item() > -1000 and C_sample.item() > constraint_threshold:  #  safety_threshold:
            break

    for j in range(num_agents):  # set-up agents
        if communication:
            if full_communication:
                n_dimensions = num_agents
                communication_indices_list = [kk for kk in range(num_agents)]
            else:
                n_dimensions = 2 if j==0 or j==num_agents-1 else 3
                communication_indices_list = [j, j + 1] if j == 0 else [j - 1, j, j + 1] if 0 < j < num_agents - 1 else [num_agents - 2, num_agents - 1]
        else:
            n_dimensions = 1
            communication_indices_list = [j]
        communication_list_dict[j] = communication_indices_list
        # X_plot needs to be determined for every agent given their position in graph
        X_plot = compute_X_plot(n_dimensions=n_dimensions, points_per_axis=int(1e4**(1/n_dimensions)))  # more points?
        # which indices are relevant for this agent? for agent 0 it is 0 and 1, for agent 1 it is 0,1,2 etc; see graph
        X_plot_dict[j] = X_plot

    for t in tqdm(range(1, iterations)):
        for j in range(num_agents):  # this is parallelizable
            if time_latent_variable:
                j, agents_j = process_agent(j, X_plot_dict[j], communication_list_dict[j], X_sample_full, Y_sample, C_sample, t, hyperparameters)
            else:
                j, agents_j = process_agent(j, X_plot_dict[j], communication_list_dict[j], X_sample_full, Y_sample, C_sample, 0, hyperparameters)
            agents[j] = agents_j  # global dict.
        # with Pool(processes=num_agents) as pool:
        #     results = pool.starmap(process_agent, [(j, agents[j], X_sample_full, Y_sample, t, hyperparameters) for j in range(num_agents)])
        # for j, agents_j in results:
        #     agents[j] = agents_j


        if t != iterations - 1:  # last iteration; do not add the new point; we just want the updated model
            if sequential_expert or random_expert:
                if sequential_expert:
                    expert_agent = (t - 1) % num_agents
                elif random_expert:
                    expert_agent = np.random.choice(range(num_agents))  # who is the expert this round?
                expert_x_new_neighbors = agents[expert_agent][1][:-1]  # exclude time. Wait time is never in?
                expert_communication_list = communication_list_dict[expert_agent]
                x_new_full = torch.zeros(num_agents).unsqueeze(0)
                x_new_full[:, expert_communication_list] = expert_x_new_neighbors
                for jj in range(num_agents):
                    if jj not in expert_communication_list:
                        x_new_full[:, jj] = agents[jj][0]  # get their x_new prediction
            else:
                if full_communication:
                    x_new_full = agents[0][0][:-1].unsqueeze(0)  # we can take any agent; they are all the same because we model all and communicate everything
                    # -1 because we leave time domain out!
                else:
                    x_new_full = torch.cat([agents[j][0] for j in range(num_agents)]).unsqueeze(0)  # concatenate all x_new ("1D ones")
            y_new = torch.tensor([float(gt.f(x_new_full))], dtype=torch.float32)  
            c_new = torch.tensor([float(gt.c(x_new_full))], dtype=torch.float32) 
            
            Y_sample = torch.cat((Y_sample, y_new), dim=0)
            C_sample = torch.cat((C_sample, c_new), dim=0)
            X_sample_full = torch.cat((X_sample_full, x_new_full))
    # Just save the cube!  # also save beta
    with open('agents_8_50_107_full_communication.pickle', 'wb') as handle:
        dill.dump(agents, handle)


    # Development of reward
    plot_reward(cube=agents[0][-1])

    plot_2D_UCB(cube_dict=agents[0][-1], agent_number=0)
    plot_2D_UCB(cube_dict=agents[num_agents-1][-1], agent_number=num_agents-1)

    plot_2D_mean(cube_dict=agents[0][-1], agent_number=0)
    plot_2D_mean(cube_dict=agents[num_agents-1][-1], agent_number=num_agents-1)



    # Agents 1 and 2 3D explored domain
    for j in range(1, num_agents-1):  # not the first, not the last
        plot_3D_sampled_space(cube_dict=agents[j][-1], agent_number=j)

    # All agents 1D explored domain
    for j in range(1, num_agents-1):
        plot_1D_sampled_space(cube_dict=agents[j][-1], agent_number=j)

    # How much did we explore? Convex hull volume
    # hull = ConvexHull(X_sample_full.numpy())
    # print(f'We explored about {hull.volume*100}% of the domain.')






'''
    for j in range(1, num_agents):  # not the first because we keep this constant
        cube = agents[j][-1]
        plt.figure()
        plt.plot(np.asarray(X_sample_full[:, j]), np.asarray(Y_sample), 'ob', markersize=10)
        # plt.plot(X_plot, gt.f(X_plot), '-b')
        plt.fill_between(X_plot.flatten(), cube.lcb.detach().numpy(), cube.ucb.detach().numpy(), color='gray', alpha=0.25)
        if safety_threshold != -np.infty:
            plt.plot(X_plot.flatten(), torch.ones_like(X_plot.flatten())*safety_threshold, '-r')
        plt.xlabel('Action')
        plt.ylabel('Reward')
        plt.title(f'Agent {j}')
        # plt.savefig(f'../{num_agents}_agents_agent_{j}_safety.png')

    # Now plot Agent 0 (the one with constant)
    plt.figure()
    plt.plot(range(iterations), Y_sample, '*-')
    plt.title(f'Agent 0: Time series POV; {num_agents} total agents, safety threshold={round(safety_threshold,2)}')
    plt.xlabel('Iteration')
    plt.ylabel('Global reward')
    # plt.savefig(f'../{num_agents}_agents_agent_0_safety.png')
'''
