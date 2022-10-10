# from lib2to3.pytree import type_repr
import sys
import os
from parso import parse
import torch.nn as nn
import torch.nn.functional as F
curr_path = os.path.dirname(os.path.abspath(__file__))  
parent_path = os.path.dirname(curr_path)  
sys.path.append(parent_path)  

import gym
import torch
import datetime
import numpy as np
import argparse
from common.utils import save_results_1, make_dir
from common.utils import plot_rewards,save_args

from mdp import QLearning

from satellite_env import SatelliteEnv

def get_args():
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='Qlearning',type=str,help="name of algorithm")
    #parser.add_argument('--env_name',default='CartPole-v1',type=str,help="name of environment")
    parser.add_argument('--env_name',default='SatelliteEnv',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=300,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=0.9,type=float,help="discounted factor")
    parser.add_argument('--epsilon_start',default=0.95,type=float,help="initial value of epsilon")
    parser.add_argument('--epsilon_end',default=0.1,type=float,help="final value of epsilon")
    parser.add_argument('--epsilon_decay',default=300,type=int,help="decay rate of epsilon")
    parser.add_argument('--lr',default=0.01,type=float,help="learning rate")
    parser.add_argument('--memory_capacity',default=100000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=64,type=int)
    parser.add_argument('--target_update',default=4,type=int)
    parser.add_argument('--hidden_dim',default=32,type=int)
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results/' )
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/models/' ) # path to save models
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")           
    args = parser.parse_args()    
    args.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # check GPU                        
    return args


def env_agent_config(cfg,seed=1):
    env = SatelliteEnv() # create env
    #n_states = env.observation_space.shape[0]  # dim of states
    n_states = env.observation_space.n  # dim of states
    n_actions = env.action_space.n  # dim of actions
    print(f"n states: {n_states}, n actions: {n_actions}")

    agent = QLearning(n_states, n_actions, cfg)  
    
    if seed !=0: # random seed
        torch.manual_seed(seed)
        #env.seed(seed)
        np.random.seed(seed)
    return env, agent


def train(cfg, env, agent):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algo: {cfg.algo_name}, Device: {cfg.device}')
    rewards = []    # total reward
    ma_rewards = []  
    steps = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # reward in one episode
        ep_step = 0
        state = env.reset()[0]  # reset env     # modify this without [0] for other env
        
        while True:
            action = agent.sample_action(state)  # choose action
            (next_state, reward, done, _, _) = env.step(action)  # update env
            agent.update(state, action, reward, next_state, done)   # update agent
            state = next_state  # store next state
            ep_reward += reward  # add up reward
            ep_step += 1
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 1 == 0:
            print(f'Episode：{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f} Epislon:{agent.epsilon:.3f}')
    print('Finish training!')
    env.close()
    res_dic = {'rewards':rewards,'ma_rewards':ma_rewards,'steps':steps}
    return res_dic





def test(cfg, env, agent):
    print('start testing!')
    print(f'environment:{cfg.env_name}, algorithm:{cfg.algo_name}, device:{cfg.device}')
    ############# no eps-greedy needed for testing ###############
    cfg.epsilon_start = 0.0  
    cfg.epsilon_end = 0.0
    agent.epsilon = 0.0  
    ################################################################################
    rewards = []  
    ma_rewards = []  
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  
        ep_step = 0
        state = env.reset()[0]      # modify for other env
        while True:
            action = agent.predict_action(state)  # choose action
            print(action)
            (next_state, reward, done, _, _) = env.step(action)  # update env
            state = next_state  # store next state
            ep_reward += reward  # add up reward
            ep_step += 1
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f'Episode：{i_ep+1}/{cfg.test_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f}, Epislon:{agent.epsilon:.3f}')
    print('finished testing!')
    env.close()
    return {'rewards':rewards,'ma_rewards':ma_rewards,'steps':steps}


if __name__ == "__main__":
    cfg = get_args()
    # training
    env, agent = env_agent_config(cfg)
    res_dic = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  
    save_args(cfg)
    agent.save(path=cfg.model_path)  
    save_results_1(res_dic, tag='train',
                 path=cfg.result_path)  
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="train")  
    # testing
    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)  
    res_dic = test(cfg, env, agent)
    save_results_1(res_dic, tag='test',
                 path=cfg.result_path)  
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'],cfg, tag="test")  
