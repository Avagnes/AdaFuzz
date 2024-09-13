if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import copy
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTester
import tensorflow.contrib.layers as layers
import tqdm, sys
import torch
from datetime import datetime
from fuzz.ada_fuzz import Tester, SeedSpace
import traceback

result_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--random-seed", help="Random seed", default=0, type=int)
    parser.add_argument("--hours", help="time limit", default=1, type=float)
    parser.add_argument("--family-size", help="initial seed number", default=10, type=int)
    parser.add_argument("--warmup-times", help="test times for warmup", default=100, type=int)
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=300000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='spread', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../checkpoints/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False, fuzz=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if fuzz:
        env = MultiAgentEnv(world, scenario.reset_world_fuzz, scenario.reward, scenario.observation,
                            scenario.benchmark_data, scenario.done_flag, verify_func=scenario.verify)
    else:
        env = MultiAgentEnv(world, scenario.reset_world_before_fuzz, scenario.reward, scenario.observation,
                            scenario.benchmark_data, scenario.done_flag)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTester
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def get_observe(env):
    state = []
    for agent in env.world.agents:
        state.append(agent.state.p_pos)
        state.append(agent.state.p_vel)
        state.append(agent.state.c)
    for landmark in env.world.landmarks:
        state.append(landmark.state.p_pos)
    return list(np.array(state).flatten())



def get_init_state(env):
    state = []
    for agent in env.world.agents:
        state.append(agent.state.p_pos)
    for landmark in env.world.landmarks:
        state.append(landmark.state.p_pos)
    return state


def get_collision_num(env):
    collisions = 0
    for i, agent in enumerate(env.world.agents):
        for j, agent_other in enumerate(env.world.agents):
            if i == j:
                continue
            delta_pos = agent.state.p_pos - agent_other.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = (agent.size + agent_other.size)
            if dist < dist_min:
                collisions += 1
    return collisions / 2


def ada_test(arglist):
    np.random.seed(arglist.random_seed)
    pbar = tqdm.tqdm(total=int(arglist.hours * 3600 * arglist.hours * 50))
    with U.single_threaded_session():
        env = make_env(arglist.scenario, arglist, arglist.benchmark, fuzz=True)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        
        U.initialize()
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        U.load_state(arglist.load_dir)
        init_state = get_init_state(env)
        seed_space = SeedSpace()
        result = []
        result_time = []
        all_seeds = []
        tester = Tester(arglist.family_size)
        mdp_test_time = 0
        try:
            with open("marl"+str(arglist.random_seed), 'rb') as f:
                normal_cases = pickle.load(f)
            normal_cases = list(np.array(normal_cases))
        except:
            normal_cases = SeedSpace.create_data()
            with open("marl"+str(arglist.random_seed), 'wb') as f:
                pickle.dump(normal_cases, f)

        start_test_time = time.time()
        while time.time() - start_test_time < 3600 * arglist.hours:
            while len(tester.families) < arglist.family_size:
                if len(normal_cases):
                    seed = normal_cases.pop(0)
                else:
                    seed = seed_space.random_generate()
                seed = list(np.array(seed))
                obs_n = env.reset(seed[0:3], seed[3:])
                agent_flag, landmark_flag = env.verify_func(env.world)
                if agent_flag or landmark_flag:
                    continue
                mdp_test_time += 1
                pbar.update(1)

                steps = 0
                episode_reward = 0
                collisions = 0
                sequence = []
                sequence.append(get_observe(env))
                all_seeds.append(tuple(seed))
                while True:
                    steps += 1
                    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                    new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                    done = all(done_n)
                    terminal = (steps >= arglist.max_episode_len)
                    sequence.append(get_observe(env))
                    collisions += get_collision_num(env)
                    obs_n = new_obs_n
                    for i, rew in enumerate(rew_n):
                        episode_reward += rew
                    if terminal and collisions > 5 and not done:
                        win = 1
                        break
                    elif done or terminal:
                        win = 0
                        break
                if win == 1:
                    result.append(tuple(seed))
                    result_time.append((time.time()-start_test_time))
                    continue
                intrinsic_reward = float(tester.train_rnd(sequence))
                samplePr = np.exp(-episode_reward*0.01)+np.exp(-intrinsic_reward)
                tester.add_family(seed, samplePr, episode_reward)
            seed, reward = tester.get_seed()
            mutate_seed = seed_space.mutate(seed)
            mdp_test_time += 1
            obs_n = env.reset(mutate_seed[0:3], mutate_seed[3:])
            agent_flag, landmark_flag = env.verify_func(env.world)
            mutate_count = 0
            while agent_flag or landmark_flag:
                mutate_count += 1
                if mutate_count > 10:
                    break
                new_pos = seed_space.mutate(seed)
                obs_n = env.reset(new_pos[0:3], new_pos[3:])
                # print(new_pos)
                agent_flag, landmark_flag = env.verify_func(env.world)
            if mutate_count > 10:
                tester.delete_family(tester.current_family)
                continue
            steps = 0
            episode_reward = 0
            collisions = 0
            sequence = []
            sequence.append(get_observe(env))
            while True:
                steps += 1
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                done = all(done_n)
                terminal = (steps >= arglist.max_episode_len)
                sequence.append(get_observe(env))
                collisions += get_collision_num(env)
                obs_n = new_obs_n
                for i, rew in enumerate(rew_n):
                    episode_reward += rew
                if terminal and collisions > 5 and not done:
                    win = 1
                    break
                elif done or terminal:
                    win = 0
                    break
            mutate_reward = episode_reward
            all_seeds.append(tuple(mutate_seed))
            if win == 1:
                result.append(tuple(mutate_seed))
                result_time.append((time.time()-start_test_time))
                tester.delete_family(tester.current_family)
            else:
                intrinsic_reward = float(tester.train_rnd(sequence))
                samplePr = np.exp(-episode_reward*0.01)+np.exp(-intrinsic_reward)
                tester.add_seed(mutate_seed, samplePr, mutate_reward)
                tester.update_threshold()
                check = np.array(tester.samplePr) < tester.threshold
                while check.any():
                    for k in range(len(check)):
                        if check[k]:
                            tester.delete_family(k)
                            break
                    check = np.array(tester.samplePr) < tester.threshold
            pbar.update(1)
            pbar.set_postfix({'Found': len(result), 'Families': tester.family_number, 'corpus': tester.seed_number,
                              'threshold': tester.threshold})
        pbar.close()
        ## print(f"use {test_time / 60} minutes {mdp_test_time} tests to find {len(result)} results with {tester.family_number} families")

    
    with open('AdaFuzz'+result_time, 'wb') as handle:
        pickle.dump((all_seeds, result, result_time), handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    arglist = parse_args()
    f = open('./results/AdaFuzz' + result_time + '.log', 'w', buffering=1)
    sys.stdout = f
    ada_test(arglist)
