import argparse
import gym
import torch
import numpy as np
from collections import namedtuple
import network_sim
import warnings
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self) -> None:
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(30, 128)

        # actor's layer
        self.action_mean = nn.Linear(128,1)
        self.action_log_var = nn.Linear(128,1)

        # critic's layer
        self.value = nn.Linear(128,1)

        self.relu = nn.ReLU()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = self.relu(self.affine1(x))

        # actor: choose action to take from state s_t by returning probability of each action
        action_mean = self.action_mean(x)

        # critic: evaluate being in the state s_t
        state_value = self.value(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_mean, state_value


def select_action(state, model):
    state = torch.from_numpy(state).float()
    action_mean, state_value = model(state)

    m = torch.distributions.Normal(action_mean, 5.0*torch.ones_like(action_mean))

    # sample an action
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # return the action value
    return action.item()


def finish_episode(args, optimizer, model):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # TODO: figure the two losses
        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(torch.nn.functional.smooth_l1_loss(value, torch.tensor([R])))
    
    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]



def evaluate_model(args, episode_num, model, save_json=False):
    env = gym.make('PccNs-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    state = env.reset(reward=args.reward, max_bw=args.bandwidth, test=True)
    ep_reward = 0
    for t in range(1, 10000):
        state = torch.from_numpy(state).float()
        action_mean, _ = model(state)
        action = action_mean.item()

        state, reward, done, _ = env.step(action)

        model.rewards.append(reward)
        ep_reward += reward
        if done:
            break
    
    if save_json:
        env.dump_events_to_file('./logs/test_rl_%s_%.2f.json' % (args.reward, args.bandwidth))

    return ep_reward


def main(args):
    env = gym.make('PccNs-v0')
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    model = Policy()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    total_test_rewards = []
    running_reward = 10

    for i_episode in range(args.episodes):
        # print('training in episode {}'.format(i_episode))

        # reset env
        state = env.reset(reward=args.reward, max_bw=args.bandwidth, test=False)
        ep_reward = 0

        # in each episode, run 10000 steps
        for _ in range(1,10000):
            # select action
            action = select_action(state, model)
            
            # take action
            state, reward, done, _ = env.step(action)

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        
        # cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform_backprop
        finish_episode(args, optimizer, model)

        # log results
        if i_episode % args.log_interval == 0:
            test_reward = evaluate_model(args, i_episode, model, save_json=False)
            total_test_rewards.append(test_reward)            
            print('Eposide {}\tLast reward: {:.2f}\tAverage reward: {:.2f}, Test reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward, test_reward))

    # log json file
    evaluate_model(args, args.episodes, model, save_json=True)
    # save the model
    torch.save(model.state_dict(), './saved_models/model_%s_bw_%.2f.pkl' % (args.reward, args.bandwidth))
    # save the total test_rewards
    torch.save(total_test_rewards, './logs/total_test_rewards_%s_bw_%.2f.pkl'  % (args.reward, args.bandwidth))



def parse_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    res = {
        'time_data': [float(event['Time']) for event in data['Events'][1:]],
        'rew_data': [float(event['Reward']) for event in data['Events'][1:]],
        'send_data': [float(event['Send Rate']) for event in data['Events'][1:]],
        'thpt_data': [float(event['Throughput']) for event in data['Events'][1:]],
        'latency_data': [float(event['Latency']) for event in data['Events'][1:]],
        'loss_data': [float(event['Loss Rate']) for event in data['Events'][1:]]
    }
    return res


def plot_latency_and_loss(test_rl_throughput):
    for bw in test_rl_throughput:
        fig, axs = plt.subplots(2)
        plt.subplots_adjust(hspace=0.4)
        axs[0].plot(test_rl_throughput[bw]['time_data'], test_rl_throughput[bw]['latency_data'])
        axs[0].set(ylabel='Latency (sec)')
        axs[0].set(title='Latency, Max Throughput = %d Mbps' % bw)
        axs[0].set(ylim=(0.0, 0.5))

        axs[1].plot(test_rl_throughput[bw]['time_data'], test_rl_throughput[bw]['loss_data'])
        axs[1].set(ylabel='Loss rate')
        axs[1].set(xlabel='Monitor Interval')
        axs[1].set(title='Loss rate, Max Throughput = %d Mbps' % bw)
        #axs[1].set(ylim=(0.0, 0.2))

        fig.savefig('./figures/latency_loss_graph_bw=%d.png' % bw)
        #print(results['loss_data'])

def plot_throughput_vs_sendrate(test_rl_throughput, test_rl_latency):
    res_throughput = []
    res_latency = []
    max_bws = []

    for bw in test_rl_throughput:
        res_throughput.append((np.array(test_rl_throughput[bw]['thpt_data'][350:])/1e6).mean())
        res_latency.append((np.array(test_rl_latency[bw]['thpt_data'][350:])/1e6).mean())
        max_bws.append(bw)

    plt.figure()
    plt.scatter(max_bws, res_throughput, c='b', label='RL-throughput')
    plt.scatter(max_bws, res_latency, c='g', label='RL-latency')
    plt.plot(max_bws, res_throughput, '-b')
    plt.plot(max_bws, res_latency, '--g')
    
    plt.plot(range(24), range(24), '-r', label='Optimal')
    plt.plot(np.ones(24)*1.2, range(24), '-k')
    plt.plot(np.ones(24)*6.0, range(24), '-k')
    plt.grid()
    plt.xlabel('Link Capacity [Mbps]')
    plt.ylabel('Achieved Throughput [Mbps')
    plt.legend()
    plt.title('Bandwidth sensitivity')
    plt.savefig('./figures/throughput_vs_sendrate.png')


def plot_comparison_with_different_rewards(test_rl_throughput, test_rl_latency):
    for bw in test_rl_throughput:
        res_throughput = test_rl_throughput[bw]
        res_latency = test_rl_latency[bw]
        fig, axs = plt.subplots(3)
        plt.subplots_adjust(hspace=0.4)
        fig.set_size_inches(11.5, 7.5)
        axs[0].plt(res_throughput['time_data'], np.array(res_throughput['thpt_data'])/1e6, label='Reward=throughput')
        axs[0].plt(test_rl_latency['time_data'], np.array(test_rl_latency['thpt_data'])/1e6, label='Reward=latency')
        axs[0].set(ylabel='Throughput [Mbps]')
        axs[0].set(title='Throughput of different rewards')
        axs[0].grid()
        axs[0].legend(loc='lower right')


        axs[1].plt(res_throughput['time_data'], res_throughput['latency_data'], label='Reward=throughput')
        axs[1].plt(test_rl_latency['time_data'], test_rl_latency['latency_data'], label='Reward=latency')
        axs[1].set(ylabel='Latency [sec]')
        axs[1].set(title='Latency of different rewards')
        axs[1].set(ylim=(0.15, 0.25))
        axs[1].grid()
        axs[1].legend(loc='lower right')

        axs[2].plt(res_throughput['time_data'], res_throughput['loss_data'], label='Reward=throughput')
        axs[2].plt(test_rl_latency['time_data'], test_rl_latency['loss_data'], label='Reward=latency')
        axs[2].set(xlabel='Monitor Interval')
        axs[2].set(ylabel='Loss rate')
        axs[2].set(title='Loss rate of different rewards')
        #axs[2].set(ylim=(0, 1))
        axs[2].grid()
        axs[2].legend(loc='lower right')

        fig.save('./figures/comparison_with_rewards_bw=%.2f.png' % bw)

def plot_test_rewards():
    # plot figures
    # usage: python main2.py --plot
    print('plot rewards')
    fig_path = './figures/'
    rewards = np.array(torch.load('total_test_rewards.pkl'))
    plt.figure()
    plt.plot(np.arange(rewards.shape[0])*10, rewards)
    plt.grid()
    plt.xlabel('Training Episodes')
    plt.ylabel('Reward')
    plt.title('Accumulated Evaluation Reward per train episode')
    plt.savefig(fig_path + 'reward_plot.png')
    #print(rewards)



def analyze_results():  
    bw_list = [2.0, 5.0, 10.0, 15.0, 20.0]
    # analyze throughput vs. bandwidth
    # usage:
    # python main2.py -bw 2
    # python main2.py -bw 5
    # python main2.py -bw 10
    # python main2.py -bw 15
    # python main2.py -bw 20
    # python main2.py --analysis True
    test_rl_throughput = dict()
    for bw in bw_list:
        file_name = './logs/test_rl_throughput_{:.2f}.json'.format(bw)
        if os.path.isfile(file_name) == False:
            print('log file {} does not exist'.format(file_name))
            continue
        test_rl_throughput[bw] = parse_json(file_name)
    plot_latency_and_loss(test_rl_throughput)


    # usage:
    # python main2.py --reward latency -bw 2
    # python main2.py --reward latency -bw 5
    # python main2.py --reward latency -bw 10
    # python main2.py --reward latency -bw 15
    # python main2.py --reward latency -bw 20
    # python main2.py --analysis True
    test_rl_latency = dict()
    for bw in bw_list:
        file_name = './logs/test_rl_latency_{:.2f}.json'.format(bw)
        if os.path.isfile(file_name) == False:
            print('log file {} does not exist'.format(file_name))
            continue
        test_rl_latency[bw] = parse_json(file_name)
    plot_throughput_vs_sendrate(test_rl_throughput, test_rl_latency)


    # compare throughput, latency, loss rate under different rewards
    plot_comparison_with_different_rewards(test_rl_throughput, test_rl_latency)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default:0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default:543)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default:10)')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--bandwidth', '-bw', type=float, default=5, help='Network bandwidth in Mbps')
    parser.add_argument('--reward', type=str, default='throughput', choices=['throughput', 'latency'], help='RL agent\'s goal')
    parser.add_argument('--plot', '-p', action="store_true", help='plot training/test rewards')
    parser.add_argument('--analysis', '-a', action="store_true", help='analyze results')

    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    if args.plot == True:
        plot_test_rewards()
    elif args.analysis == True:
        analyze_results()
    else:
        main(args)


    