import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from pacsbo.pacsbo_main import compute_X_plot
# import tikzplotlib
import dill


def plot_2D_mean(cube_dict, agent_number, save=False):
    t = cube_dict['iteration'].item()
    n_dimensions = cube_dict['x_sample'].shape[1] - 1  #fixed in new version
    discr_domain = compute_X_plot(n_dimensions=n_dimensions, points_per_axis=int(1e4**(1/n_dimensions)))
    plt.figure()
    m = cube_dict['mean'].detach().numpy()
    sc = plt.scatter(
        discr_domain[:, 0],
        discr_domain[:, 1],
        c=m,
        cmap='plasma'
    )
    plt.colorbar(sc, label="Mean Value")  # Add a colorbar to show the mapping
    plt.title(f'Iteration {t}; Agent {agent_number} mean')
    plt.scatter(cube_dict['x_sample'][:, 0], cube_dict['x_sample'][:, 1], label="Sampled Points", color='k')
    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')
    if not save:
        plt.show()
    else:
        plt.savefig(f'mean_value_agent_{agent_number}.png')
        # tikzplotlib.save(f'mean_value_agent_{agent_number}.tex')


def plot_2D_UCB(cube_dict, agent_number, save=False):
    t = cube_dict['iteration'].item()
    n_dimensions = cube_dict['x_sample'].shape[1] - 1  #fixed in new version
    discr_domain = compute_X_plot(n_dimensions=n_dimensions, points_per_axis=int(1e4**(1/n_dimensions)))
    plt.figure()
    ucb = cube_dict['mean'].detach().numpy() + cube_dict['var'].detach().numpy()*cube_dict['beta'].detach().numpy()
    sc = plt.scatter(
        discr_domain[:, 0],
        discr_domain[:, 1],
        c=ucb,
        cmap='plasma'
    )
    cbar = plt.colorbar(sc)  # , label="Mean Value"  # Add a colorbar to show the mapping
    num_ticks = 5  # Set the number of ticks
    tick_positions = np.linspace(ucb.min(), ucb.max(), num_ticks)
    rounded_labels = np.round(tick_positions, 2)  # Round to 2 decimal places

    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(rounded_labels)  # Apply rounded labels
    # plt.title(f'Iteration {t}; Agent {agent_number} mean')
    plt.scatter(cube_dict['x_sample'][:, 0], cube_dict['x_sample'][:, 1], color='k')
    plt.scatter(cube_dict['x_sample'][0, 0], cube_dict['x_sample'][0, 1], color='white')
    # plt.xlabel('$a_1$')
    # plt.ylabel('$a_2$')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.gca().set_frame_on(False)
    plt.axis('off')

    plt.xlim(torch.tensor(-0.03), torch.tensor(1.03))
    plt.ylim(torch.tensor(-0.03), torch.tensor(1.03))

    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])

    if not save:
        plt.show()
    else:
        plt.savefig(f'UCB_agent_{agent_number}.png')
        plt.savefig(f'UCB_agent{agent_number}.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)




def plot_2D_samples(cube_dict, agent_number, save=False):
    t = cube_dict['iteration'].item()
    x_sample = cube_dict['x_sample']
    # discr_domain = compute_X_plot(n_dimensions=n_dimensions, points_per_axis=int(1e4**(1/n_dimensions)))
    plt.figure()
    m = cube_dict['y_sample'].detach().numpy()
    sc = plt.scatter(
        x_sample[:, 0],
        x_sample[:, 1],
        c=m,
        cmap='plasma'
    )
    plt.colorbar(sc, label="Sample")  # Add a colorbar to show the mapping
    plt.title(f'Iteration {t}; Agent {agent_number} samples')
    plt.scatter(cube_dict['x_sample'][:, 0], cube_dict['x_sample'][:, 1], label="Sampled Points", color='k')
    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')

    if not save:
        plt.show()
    else:
        plt.savefig(f'samples_2D_agent{agent_number}.png')




def plot_reward(cube, save=False):
    plt.figure()
    X_sample = cube['x_sample']
    Y_sample = cube['y_sample']
    safety_threshold = cube['safety_threshold']
    plt.plot(range(len(X_sample)), Y_sample.detach().numpy(), '-*', label='Samples')
    # plt.plot(range(len(X_sample)), torch.ones(len(X_sample))*max(gt.fX), '--g', label='Max')
    if safety_threshold > -np.inf:
        plt.plot(range(len(X_sample)), torch.ones(len(X_sample))*safety_threshold, '-r', label='Safety threshold')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    # plt.legend()
    if not save:
        plt.show()
    else:
        plt.savefig('reward_development.png')
        # tikzplotlib.save('reward_development.tex')


def plot_3D_sampled_space(cube_dict, agent_number, save=False):
    X_sample = cube_dict['x_sample'][:, :-1]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 0.9, 0.9], projection='3d')
    ax.scatter(X_sample[:, 0], X_sample[:, 1], X_sample[:, 2], color='black', s=100)
    # Labels
    ax.set_xlabel('$a_1$')
    ax.set_ylabel('$a_2$')
    ax.set_zlabel('$a_3$')
    ax.set_title(f'Sampled space Agent {agent_number}')
    if not save:
        plt.show()
    else:
        plt.savefig(f'3D_sampled_space_agent_{agent_number}.png')
        # tikzplotlib.save(f'3D_sampled_space_agent_{agent_number}.tex')


def plot_1D_sampled_space(cube_dict, agent_number, communication=True, save=False):
    if communication:
        index = 1 if agent_number != 0 else 0
    else:
        index = 0
    X_sample = cube_dict['x_sample'][:, index]
    plt.figure()
    plt.plot()
    plt.scatter(X_sample, cube_dict['y_sample'])
    plt.xlabel(f'Parameter $a$ of Agent {agent_number}')
    plt.ylabel('Corresponding $y$ value')
    plt.title(f'1D explored domain Agent {agent_number}')
    plt.xlim([-0.1, 1.1])
    if not save:
        plt.show()
    else:
        plt.savefig(f'1D_sampled_space_agent_{agent_number}.png')
        # tikzplotlib.save(f'1D_sampled_space_agent_{agent_number}.tex')
    



if __name__ == '__main__':
    with open('agents_8_50_107_full_communication.pickle', 'rb') as handle:
        agents = dill.load(handle)
    plot_2D_UCB(cube_dict=agents[0][-1], agent_number=0)
    plot_2D_UCB(cube_dict=agents[7][-1], agent_number=7)

    plot_reward(cube=agents[0][-1])

    plot_2D_mean(cube_dict=agents[0][-1], agent_number=0)
    plot_2D_mean(cube_dict=agents[7][-1], agent_number=3)


    plot_2D_samples(cube_dict=agents[0][-1], agent_number=0)
    plot_2D_samples(cube_dict=agents[7][-1], agent_number=3)


    # Agents 1 and 2 3D explored domain
    for j in range(1, 7):  # not the first, not the last
        plot_3D_sampled_space(cube_dict=agents[j][-1], agent_number=j)

    # All agents 1D explored domain
    for j in range(1, 7):
        plot_1D_sampled_space(cube_dict=agents[j][-1], agent_number=j)
