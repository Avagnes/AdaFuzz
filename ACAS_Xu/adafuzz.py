if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from models.load_model import read_onnx  # , load_repair
import torch, time, pickle, copy, argparse, sys, tqdm
from fuzz.ada_fuzz import AdaFuzz, SeedSpace
import numpy as np
import os
from datetime import datetime
result_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


class ACASagent:
    def __init__(self, acas_speed):
        self.x = 0
        self.y = 0
        self.theta = np.pi / 2
        self.speed = acas_speed
        self.interval = 0.1
        self.model_1 = read_onnx(1, 2)
        self.model_2 = read_onnx(2, 2)
        self.model_3 = read_onnx(3, 2)
        self.model_4 = read_onnx(4, 2)
        self.model_5 = read_onnx(5, 2)
        self.prev_action = 0
        self.current_active = None

    def step(self, action):
        if action == 1:
            self.theta = self.theta + 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 2:
            self.theta = self.theta - 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 3:
            self.theta = self.theta + 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 4:
            self.theta = self.theta - 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        self.x = self.x + self.speed * np.cos(self.theta) * self.interval
        self.y = self.y + self.speed * np.sin(self.theta) * self.interval

    def act(self, inputs):
        inputs = torch.Tensor(inputs)
        # action = np.random.randint(5)
        if self.prev_action == 0:
            model = self.model_1
        elif self.prev_action == 1:
            model = self.model_2
        elif self.prev_action == 2:
            model = self.model_3
        elif self.prev_action == 3:
            model = self.model_4
        elif self.prev_action == 4:
            model = self.model_5
        action, active = model(inputs)
        # action = model(inputs)
        self.current_active = [action.clone().detach().numpy(), active.clone().detach().numpy()]
        action = action.argmin()
        self.prev_action = action
        return action

    def act_proof(self, direction):
        return direction


class Autoagent:
    def __init__(self, x, y, auto_theta, speed=None):
        self.x = x
        self.y = y
        self.theta = auto_theta
        self.speed = speed
        self.interval = 0.1

    def step(self, action):
        if action == 1:
            self.theta = self.theta + 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 2:
            self.theta = self.theta - 1.5 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 3:
            self.theta = self.theta + 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        elif action == 4:
            self.theta = self.theta - 3 / 180 * np.pi * self.interval
            while self.theta > np.pi:
                self.theta -= np.pi * 2
            while self.theta < -np.pi:
                self.theta += np.pi * 2
        self.x = self.x + self.speed * np.cos(self.theta) * self.interval
        self.y = self.y + self.speed * np.sin(self.theta) * self.interval
    
    def act(self):
        # action = np.random.randint(5)
        action = 0
        return action

class env:
    def __init__(self, acas_speed, x2, y2, auto_theta):
        self.ownship = ACASagent(acas_speed)
        self.inturder = Autoagent(x2, y2, auto_theta)
        self.row = np.linalg.norm([self.ownship.x - self.inturder.x, self.ownship.y - self.inturder.y])
        if self.inturder.x - self.ownship.x > 0:
            self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        else:
            self.alpha = np.pi - np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        # if self.inturder.x - self.ownship.x < 0:
        #     self.alpha = np.pi - self.alpha
        while self.alpha > np.pi:
            self.alpha -= np.pi * 2
        while self.alpha < -np.pi:
            self.alpha += np.pi * 2
        self.phi = self.inturder.theta - self.ownship.theta
        while self.phi > np.pi:
            self.phi -= np.pi * 2
        while self.phi < -np.pi:
            self.phi += np.pi * 2

        if x2 == 0:
            if y2 > 0:
                self.inturder.speed = self.ownship.speed / 2
            else:
                self.inturder.speed = np.min([self.ownship.speed * 2, 1600])
        elif self.ownship.theta == self.inturder.theta:
            self.inturder.speed = self.ownship.speed
        else:
            self.inturder.speed = self.ownship.speed * np.sin(self.alpha) / np.sin(self.alpha + self.ownship.theta - self.inturder.theta)

        if self.inturder.speed < 0:
            self.inturder.theta = self.inturder.theta + np.pi
            while self.inturder.theta > np.pi:
                self.inturder.theta -= 2 * np.pi
            self.inturder.speed = -self.inturder.speed
        self.Vown = self.ownship.speed
        self.Vint = self.inturder.speed

    def update_params(self):
        self.row = np.linalg.norm([self.ownship.x - self.inturder.x, self.ownship.y - self.inturder.y])
        self.Vown = self.ownship.speed
        self.Vint = self.inturder.speed
        self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row)
        if self.inturder.x - self.ownship.x > 0:
            self.alpha = np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta
        else:
            self.alpha = np.pi - np.arcsin((self.inturder.y - self.ownship.y) / self.row) - self.ownship.theta

        while self.alpha > np.pi:
            self.alpha -= np.pi * 2
        while self.alpha < -np.pi:
            self.alpha += np.pi * 2
        self.phi = self.inturder.theta - self.ownship.theta
        while self.phi > np.pi:
            self.phi -= np.pi * 2
        while self.phi < -np.pi:
            self.phi += np.pi * 2

    def step(self):
        acas_act = self.ownship.act([self.row, self.alpha, self.phi, self.Vown, self.Vint])
        auto_act = self.inturder.act()
        self.ownship.step(acas_act)
        self.inturder.step(auto_act)
        self.update_params()
        # time.sleep(0.1)

    def step_proof(self, direction):
        acas_act = self.ownship.act_proof(direction)
        auto_act = self.inturder.act()
        self.ownship.step(acas_act)
        self.inturder.step(auto_act)
        self.update_params()


def verify(acas_speed, x2, y2, auto_theta):
    dis_threshold = 200
    air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air1.update_params()

    air2 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air2.update_params()

    min_dis1 = air1.row
    min_dis2 = air2.row
    for j in range(100):
        air1.step_proof(3)
        if min_dis1 > air1.row:
            min_dis1 = air1.row
    for j in range(100):
        air2.step_proof(4)
        if min_dis2 > air2.row:
            min_dis2 = air2.row
    # print(min_dis1, min_dis2)
    if (air1.Vint <= 1200 and air1.Vint >= 0) and (min_dis1 >= dis_threshold or min_dis2 >= dis_threshold):
        return True
    else:
        return False

def normalize_state(x):
    y = copy.deepcopy(x)
    y = np.array(y)
    y = y - np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
    y = y / np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])
    return y.tolist()


def get_state(acas_speed, x2, y2, auto_theta):
    air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air1.update_params()
    return normalize_state([air1.row, air1.alpha, air1.phi, air1.Vown, air1.Vint])

def reward_func(acas_speed, x2, y2, auto_theta):
    dis_threshold = 200
    air1 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
    air1.update_params()
    gamma = 0.99
    min_dis1 = np.inf
    reward = 0
    collide_flag = False
    states_seq = []

    for j in range(100):
        air1.step()
        reward = reward * gamma + air1.row / 60261.0
        states_seq.append(normalize_state([air1.row, air1.alpha, air1.phi, air1.Vown, air1.Vint]))
        if air1.row < dis_threshold:
            collide_flag = True
            reward -= 100

    return reward, collide_flag, states_seq


def fuzz_testing(fuzzer: AdaFuzz, args):  
    result = []
    all_seeds = []
    result_time = []
    np.random.seed(args.seed)
    seed_space = SeedSpace()
    try:
        with open("acas"+str(args.seed), 'rb') as f:
            normal_cases = pickle.load(f)
    except:
        normal_cases = SeedSpace.create_data()
        with open("acas"+str(args.seed), 'wb') as f:
            pickle.dump(normal_cases, f)
    def add_family():
        nonlocal result, fuzzer, seed_space, result_time, normal_cases, pbar
        while len(fuzzer.families) < args.family_size:
            if len(normal_cases):
                seed = SeedSpace.map_normal_case(normal_cases.pop(0))
            else:
                seed = seed_space.random_generate()
            if not verify(*seed):
                continue
            pbar.update(1)
            all_seeds.append(seed)
            reward, collide_flag, states_seq = reward_func(*seed)
            if collide_flag:
                result.append(seed)
                result_time.append(time.time())
                continue
            intrinsic_reward = fuzzer.train_rnd(states_seq)
            samplePr = intrinsic_reward + np.exp(np.round(-reward)*0.1)
            fuzzer.add_family(seed, samplePr, reward)
    start_fuzz_time = time.time()
    pbar = tqdm.tqdm(total=3600 * args.terminate * 50)
    test_times = 0
    while time.time() - start_fuzz_time < args.terminate * 3600:
        add_family()
        seed, reward = fuzzer.get_seed()
        mutate_seed = seed_space.mutate(seed)
        break_flag = 0
        while verify(*mutate_seed) == False:
            break_flag += 1
            mutate_seed = seed_space.mutate(seed)
            if break_flag > 50:
                break
        if break_flag > 50:
            fuzzer.delete_family()
            continue
        all_seeds.append(mutate_seed)
        mutate_reward, collide_flag, mutated_states_seq = reward_func(*mutate_seed)
        test_times = test_times + 1
        if collide_flag:
            result.append(mutate_seed)
            result_time.append(time.time())
            print('found: ', len(result))
            fuzzer.delete_family(fuzzer.current_family)
        else:
            intrinsic_reward = fuzzer.train_rnd(mutated_states_seq)
            samplePr = intrinsic_reward + np.exp(np.round(-mutate_reward)*0.1)
            fuzzer.add_seed(mutate_seed, samplePr, mutate_reward)
            fuzzer.update_threshold()
            check = np.array(fuzzer.samplePr) < fuzzer.threshold
            while check.any():
                for k in range(len(check)):
                    if check[k]:
                        fuzzer.delete_family(k)
                        break
                check = np.array(fuzzer.samplePr) < fuzzer.threshold
        pbar.update(1)
        pbar.set_postfix({'Found': len(result), 'Families': fuzzer.family_number, 'corpus': fuzzer.seed_number, 'threshold': fuzzer.threshold})

    print('total time: ', time.time() - start_fuzz_time)
    with open('AdaFuzz'+result_time, 'wb') as handle:
        pickle.dump((all_seeds, result, (np.array(result_time)-start_fuzz_time)/3600), handle)

def replay(pickle_path):
    with open(pickle_path, 'rb') as handle:
        all_seeds, result, result_time = pickle.load(handle)

    for i in range(len(result)):
        for m in range(3):
            if m == 0:
                X1 = []
                Y1 = []
                X2 = []
                Y2 = []
                [acas_speed, x2, y2, auto_theta] = result[i]
                air = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
                air.update_params()
                min_row = air.row
                print('Vint: ', air.Vint)
                for j in range(100):
                    X1.append(air.ownship.x)
                    Y1.append(air.ownship.y)
                    X2.append(air.inturder.x)
                    Y2.append(air.inturder.y)
                    air.step()
                    if min_row > air.row:
                        min_row = air.row
                print('Min distance: ', min_row)
                min_x = np.min([np.min(X1), np.min(X2)])
                min_x -= 100
                max_x = np.max([np.max(X1), np.max(X2)])
                max_x += 100
                min_y = np.min([np.min(Y1), np.min(Y2)])
                min_y -= 100
                max_y = np.max([np.max(Y1), np.max(Y2)])
                max_y += 100
                fig = plt.figure()
                ax = plt.axes(xlim=(min_x, max_x), ylim=(min_y, max_y))
                line1, = ax.plot([], [], lw=2, c='red', label='model-controlled agent')
                line2, = ax.plot([], [], lw=2, c='blue', label='intruder')

                def init():
                    line1.set_data([], [])
                    line2.set_data([], [])
                    return line1, line2
                def animate(k):
                    line1.set_data(X1[:k], Y1[:k])
                    line2.set_data(X2[:k], Y2[:k])
                    return line1, line2
                ani = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(X1), interval=1, blit=True)
                gif_name = './results/AdaFuzz'+result_time + '_crash_' + str(i) + '_' + str(min_row) + '.gif'
                ani.save(gif_name)
            else:
                pass
                X1_3 = []
                Y1_3 = []
                X2_3 = []
                Y2_3 = []
                X1_4 = []
                Y1_4 = []
                X2_4 = []
                Y2_4 = []

                [acas_speed, x2, y2, auto_theta] = result[i]
                air3 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
                air4 = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
                air3.update_params()
                air4.update_params()

                min_row_3 = air3.row
                min_row_4 = air4.row
                for j in range(100):
                    X1_3.append(air3.ownship.x)
                    Y1_3.append(air3.ownship.y)
                    X2_3.append(air3.inturder.x)
                    Y2_3.append(air3.inturder.y)
                    X1_4.append(air4.ownship.x)
                    Y1_4.append(air4.ownship.y)
                    X2_4.append(air4.inturder.x)
                    Y2_4.append(air4.inturder.y)
                    air3.step_proof(3)
                    air4.step_proof(4)
                    if min_row_3 > air3.row:
                        min_row_3 = air3.row
                    if min_row_4 > air4.row:
                        min_row_4 = air4.row
                if min_row_3 > min_row_4:
                    X1 = X1_3
                    Y1 = Y1_3
                    X2 = X2_3
                    Y2 = Y2_3
                    min_row = min_row_3
                else:
                    X1 = X1_4
                    Y1 = Y1_4
                    X2 = X2_4
                    Y2 = Y2_4
                    min_row = min_row_4

                min_x = np.min([np.min(X1), np.min(X2)])
                min_x -= 100
                max_x = np.max([np.max(X1), np.max(X2)])
                max_x += 100
                min_y = np.min([np.min(Y1), np.min(Y2)])
                min_y -= 100
                max_y = np.max([np.max(Y1), np.max(Y2)])
                max_y += 100
                fig = plt.figure()
                ax = plt.axes(xlim=(min_x, max_x), ylim=(min_y, max_y))
                line1, = ax.plot([], [], lw=2, c='red')
                line2, = ax.plot([], [], lw=2, c='blue')

                def init():
                    line1.set_data([], [])
                    line2.set_data([], [])
                    return line1, line2
                def animate(k):
                    line1.set_data(X1[:k], Y1[:k])
                    line2.set_data(X2[:k], Y2[:k])
                    return line1, line2
                ani = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=len(X1), interval=1, blit=True)
                gif_name = './results/AdaFuzz'+result_time+ '_nocrash_' + str(i) + '_' + str(min_row) + '.gif'
                ani.save(gif_name)

def replay_repair(pickle_path, render=False):
    with open(pickle_path, 'rb') as handle:
        all_seeds, result, result_time = pickle.load(handle)
    count_nocrash = 0
    count_crash = 0
    dis_threshold = 200
    for i in range(len(result)):
        X1 = []
        Y1 = []
        X2 = []
        Y2 = []
        [acas_speed, x2, y2, auto_theta] = result[i]
        air = env(acas_speed=acas_speed, x2=x2, y2=y2, auto_theta=auto_theta)
        air.update_params()
        min_row = air.row
        for j in range(100):
            X1.append(air.ownship.x)
            Y1.append(air.ownship.y)
            X2.append(air.inturder.x)
            Y2.append(air.inturder.y)
            air.step()
            if min_row > air.row:
                min_row = air.row
        if min_row < dis_threshold:
            count_crash += 1
        else:
            count_nocrash += 1
        print('Min distance: ', min_row)
        # print(temp)
        if render:
            min_x = np.min([np.min(X1), np.min(X2)])
            min_x -= 100
            max_x = np.max([np.max(X1), np.max(X2)])
            max_x += 100
            min_y = np.min([np.min(Y1), np.min(Y2)])
            min_y -= 100
            max_y = np.max([np.max(Y1), np.max(Y2)])
            max_y += 100
            fig = plt.figure()
            ax = plt.axes(xlim=(min_x, max_x), ylim=(min_y, max_y))
            line1, = ax.plot([], [], lw=2, c='red')
            line2, = ax.plot([], [], lw=2, c='blue')

            def init():
                line1.set_data([], [])
                line2.set_data([], [])
                return line1, line2
            def animate(k):
                line1.set_data(X1[:k], Y1[:k])
                line2.set_data(X2[:k], Y2[:k])
                return line1, line2
            ani = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(X1), interval=1, blit=True)
            gif_name = './results/AdaFuzz'+result_time+'_' +  str(i) + str(min_row) + '.gif'
            # ani = animation.ArtistAnimation(fig, tmp, interval=1, repeat_delay=0, blit=True)
            ani.save(gif_name)
    print('crash: ', count_crash, ', no crash: ', count_nocrash)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--terminate", type=float, default=1)
    parser.add_argument("--picklepath", type=str, default='')
    parser.add_argument("--family-size", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)


    args = parser.parse_args()

    if args.replay and args.repair:
        # replay(args.picklepath)
        replay_repair(args.picklepath, args.render)
    elif args.replay:
        replay(args.picklepath)
    else:
        f = open('./results/AdaFuzz'+result_time+'.log', 'w', buffering=1)
        sys.stdout = f
        fuzzer = AdaFuzz(args.family_size)
        fuzz_testing(fuzzer, args)
    print('success')
