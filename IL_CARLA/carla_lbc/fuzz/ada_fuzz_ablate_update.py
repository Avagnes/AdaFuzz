import numpy as np
import copy
import carla
import torch.nn as nn
import torch
import traceback
from argparse import Namespace
import tqdm

class RND(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, output_size=16):
        super(RND, self).__init__()
        self.target_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.predictor_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)
        )

        # Initialize the target network with random weights
        for m in self.target_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)

        # Make target network's parameters not trainable
        for param in self.target_net.parameters():
            param.requires_grad = False
            
            
    def forward(self, x):
        target_out = self.target_net(x)
        predictor_out = self.predictor_net(x)
        return target_out, predictor_out

class AdaFuzz:
    class Family:
        def __init__(self, seed, entropy, reward, corpus_ratio):
            self.corpus = [seed]
            self.entropy = [entropy]
            self.rewards = [reward]
            self.childrenIndex = [[]]
            self.parentIndex = [0]
            self.current_index = 0
            self.corpus_ratio = corpus_ratio

        def get_seed(self):
            choose_index = \
                np.random.choice(range(len(self.corpus)), 1, p=np.array(self.entropy) / np.sum(self.entropy))[0]
            self.current_index = choose_index
            return self.corpus[choose_index], self.rewards[choose_index]

        def add_seed(self, seed, entropy, reward):
            self.corpus.append(seed)
            self.entropy.append(copy.deepcopy(entropy))
            self.childrenIndex.append([])
            self.parentIndex.append(self.current_index)
            self.rewards.append(reward)
            # self.update_entropy(self.current_index, len(self.entropy) - 1)

        def update_entropy(self, index, from_index):
            if from_index not in self.childrenIndex[index] and from_index != self.parentIndex[index]:
                self.childrenIndex[index].append(from_index)
            if index != 0:
                val = (self.entropy[index] + self.entropy[self.parentIndex[index]] +
                       np.array(self.entropy)[self.childrenIndex[index]].sum()) / (2 + len(self.childrenIndex[index]))
            else:
                val = (self.entropy[index] + np.array(self.entropy)[self.childrenIndex[index]].sum()) / (
                        1 + len(self.childrenIndex[index]))
            if val < 0:
                print('\033[31mval is negative\033[0m')
                val = 0
            self.entropy[index] = val

            neighbors = copy.deepcopy(self.childrenIndex[index])
            if index != 0:
                neighbors.append(self.parentIndex[index])
            for neighbor in neighbors:
                if neighbor == from_index:
                    continue
                self.update_entropy(neighbor, index)

        def represent_entropy(self):
            # tmp = (np.max(self.entropy)*(1-0.5*np.log(len(self.corpus)/self.corpus_ratio)))
            # return (0 if tmp < 0 else tmp)
            return np.max(self.entropy)

    def __init__(self, family_size, delay_factor=0, corpus_factor=0.5, corpus_ratio=10):
        self.family_size = family_size
        self.corpus_ratio = corpus_ratio
        self.corpus_factor = corpus_factor
        self.delay_factor = delay_factor
        self.families = []
        self.samplePr = []
        self.threshold = 0
        self.current_family = None
        self.seed_number = 0
        self.family_number = 0

        self.rnd = RND()
        self.rnd_standard = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rnd = self.rnd.to(self.device)
        self.optimizer = torch.optim.Adam(list(self.rnd.predictor_net.parameters()), lr=1e-7)

    def add_family(self, seed, entropy, reward):
        pose = seed[0]
        newpose = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
        vehicle_info = seed[1]
        new_vehicle_info = []
        for i in range(len(vehicle_info)):
            pose = vehicle_info[i][1]
            v_1 = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
            temp = (vehicle_info[i][0], v_1, vehicle_info[i][2], vehicle_info[i][3])
            new_vehicle_info.append(temp)
        # copy_pose = (newpose, new_vehicle_info)
        # copy_envsetting = []
        # for i in range(len(seed[2])):
        #     copy_envsetting.append(seed[2][i])
        self.families.append(self.Family((newpose, new_vehicle_info, seed[2]), entropy, reward, self.corpus_ratio))
        self.samplePr.append(entropy)
        self.seed_number += 1
        self.family_number += 1

    def get_pose(self):
        family_index = \
            np.random.choice(range(len(self.families)), 1, p=np.array(self.samplePr) / np.sum(self.samplePr))[0]
        self.current_family = family_index
        seed = self.families[family_index].get_seed()[0]
        self.current_pose = seed[0]
        self.current_vehicle_info = seed[1]
        self.current_envsetting = seed[2]
        return self.current_pose

    def add_seed(self, seed, entropy, reward):
        pose = seed[0]
        newpose = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
        vehicle_info = seed[1]
        new_vehicle_info = []
        for i in range(len(vehicle_info)):
            pose = vehicle_info[i][1]
            v_1 = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
            temp = (vehicle_info[i][0], v_1, vehicle_info[i][2], vehicle_info[i][3])
            new_vehicle_info.append(temp)
        self.families[self.current_family].add_seed((newpose, new_vehicle_info, seed[2]), entropy, reward)
        self.samplePr[self.current_family] = self.families[self.current_family].represent_entropy()
        self.seed_number += 1

    def delete_family(self, family_index):
        family = self.families.pop(family_index)
        self.samplePr.pop(family_index)
        self.seed_number -= len(family.corpus)
        del family
        self.current_family = None

    def update_threshold(self):
        seed_sum = 0
        seed_len = 0
        for family in self.families:
            seed_sum += np.sum(family.entropy)
            seed_len += len(family.entropy)
        tmp = (self.seed_number / self.family_size - 1) / (self.corpus_ratio - 1)
        self.threshold = (self.delay_factor * self.threshold + (1 - self.delay_factor) *
                          seed_sum / seed_len * (1 + self.corpus_factor * np.log(tmp)))
    
    def train_rnd(self, states, intrinsic_reward_scale=1e-6, l2_reg_coeff=1e-6):
        states = np.array(states)
        states[np.isnan(states)] = 0
        state_tensor = torch.FloatTensor(states).to(self.device)
        target_out, predictor_out = self.rnd(state_tensor)
        intrinsic_reward = torch.pow(target_out - predictor_out, 2).sum(dim=1, keepdim=True) * intrinsic_reward_scale
        # print("intrinsic: ", intrinsic_reward)
        loss = torch.mean(intrinsic_reward)
        # Add L2 regularization to the loss
        l2_reg = 0
        for param in self.rnd.predictor_net.parameters():
            l2_reg += torch.norm(param)
        loss = loss + l2_reg_coeff * l2_reg
        # print("loss", loss)
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.rnd.predictor_net.parameters(), 5)
            self.optimizer.step()
        return loss.item()

    def get_vehicle_info(self):
        return self.current_vehicle_info
    
    def mutation(self, pose):
        newpose = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
        newpose.location.x = pose.location.x + np.random.uniform(-0.15, 0.15)
        newpose.location.y = pose.location.y + np.random.uniform(-0.15, 0.15)
        newpose.rotation.yaw = pose.rotation.yaw + np.random.uniform(-5, 5)
        self.current_pose = newpose
        return newpose
    
    def vehicle_mutate(self, vehicle_info):
        new_vehicle_info = []
        for i in range(len(vehicle_info)):
            pose = vehicle_info[i][1]
            v_1 = carla.Transform(carla.Location(x=pose.location.x, y=pose.location.y, z=pose.location.z), carla.Rotation(pitch=pose.rotation.pitch, yaw=pose.rotation.yaw, roll=pose.rotation.roll))
            v_1.location.x += np.random.uniform(-0.1, 0.1)
            v_1.location.y += np.random.uniform(-0.1, 0.1)
            temp = (vehicle_info[i][0], v_1, vehicle_info[i][2], vehicle_info[i][3])
            new_vehicle_info.append(temp)

        self.current_vehicle_info = new_vehicle_info
        return self.current_vehicle_info
    

class SeedSpace:
    def random_generate():
        test_settings = Namespace(
            min = -1,
            max = 1,
            start_scope = 101,
            yaw_scope = 5,
            weather_scope = 13,
            target_scope = 101
        )
        vector_info = np.random.uniform(-1,1, 1 + 3 + 1 + 1 + 2 * 100)
        vector_info = np.clip(vector_info, test_settings.min, test_settings.max)
        test_settings.start_pose = int(((vector_info[0] -  test_settings.min) /  (test_settings.max -  test_settings.min)) * test_settings.start_scope)
        test_settings.start_pose = np.clip(test_settings.start_pose, 0, test_settings.start_scope - 1)

        if test_settings.start_pose in [39,40,41,42,43,48,51,68,79]:
            test_settings.start_pose = 1

        test_settings.start_pose_x = vector_info[1]
        test_settings.start_pose_y = vector_info[2]
        test_settings.start_pose_yaw = vector_info[3] * test_settings.yaw_scope

        test_settings.target_pose = int(((vector_info[4] -  test_settings.min) /  (test_settings.max -  test_settings.min)) * test_settings.target_scope)
        test_settings.target_pose = np.clip(test_settings.target_pose, 0, test_settings.target_scope - 1)

        test_settings.weather = int(((vector_info[5] -  test_settings.min) /  (test_settings.max -  test_settings.min)) * test_settings.weather_scope)
        
        test_settings.vehicles = []
        for i in range(test_settings.target_scope - 1):
            v_x = vector_info[6 + i * 2]
            v_y = vector_info[7 + i * 2] 
            test_settings.vehicles.append((v_x, v_y))
        return test_settings

    def create_data(div=100):
        data = np.random.uniform(-1,1,size=(50000, 1 + 3 + 1 + 1 + 2 * 100)).tolist()
        print("Begin adaptive sort!")
        normal_cases = SeedSpace._adaptive_sort(data)
        return normal_cases

    def _adaptive_sort(candidates: list):
        np.random.seed()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16
        first_idx = np.random.randint(len(candidates))

        LENGTH = len(candidates[0])
        MID = int(len(candidates)/2)
        chosen_desk = torch.zeros(len(candidates), LENGTH, dtype=dtype, device=device)
        chosen_desk[0,:] = torch.tensor(np.array([candidates.pop(first_idx)]))
        chosen = chosen_desk[:1,:]
        for_chosen = torch.tensor(np.array(candidates), dtype=dtype, device=device)
        for_chosen_desk = torch.zeros(for_chosen.shape, dtype=dtype, device=device)
        minus_desk = torch.zeros((len(candidates), LENGTH), dtype=dtype, device=device)
        value_desk = torch.zeros((len(candidates), MID), dtype=dtype, device=device)
        value_desk_desk = torch.zeros((len(candidates), MID), dtype=dtype, device=device)
        
        pbar = tqdm.tqdm(total=len(for_chosen))
        for i in range(len(candidates)-1):
            if i == MID:
                del value_desk_desk
                value_desk_desk = torch.zeros((value_desk.shape[0], len(candidates)), dtype=dtype, device=device)
                value_desk_desk[:, :value_desk.shape[1]] = value_desk
                del value_desk
                value_desk = torch.clone(value_desk_desk)
            minus_desk[:,:] = for_chosen - chosen[-1,:]
            value_desk[:,i] = torch.square(minus_desk).sum(axis=1)
            value = value_desk[:,:i+1].min(axis=1)[0]
            idx = torch.argmax(value)
            chosen_desk[len(chosen),:] = for_chosen[idx,:]
            chosen = chosen_desk[:len(chosen)+1,:]
            for_chosen_desk[:len(for_chosen)-idx-1,:] = for_chosen[idx+1:,:]
            for_chosen = for_chosen[:-1,:]
            for_chosen[idx:,:] = for_chosen_desk[:len(for_chosen)-idx,:]
            value_desk_desk[:len(value_desk)-idx-1,:] = value_desk[idx+1:,:]
            value_desk = value_desk[:-1,:]
            value_desk[idx:,:] = value_desk_desk[:len(value_desk)-idx,:]
            minus_desk = minus_desk[:-1,:]
            pbar.update(1)

        pbar.close()
        return chosen.cpu().numpy().tolist()


    def _create_data(desk: list, depth: int, div: int, tmp: list):
        if depth == 0:
            desk.append(copy.deepcopy(tmp))
            return 
        for k in range(div):
            tmp.append(2.0 / (div-1) * k - 1)
            SeedSpace._create_data(desk, depth-1, div, tmp)
            tmp.pop()

    def map_normal_case(vector_info): 
        test_settings = Namespace(
            min = -1,
            max = 1,
            start_scope = 101,
            yaw_scope = 5,
            weather_scope = 13,
            target_scope = 101
        )
        vector_info = np.clip(vector_info, test_settings.min, test_settings.max)
        test_settings.start_pose = int(((vector_info[0] -  test_settings.min) /  (test_settings.max -  test_settings.min)) * test_settings.start_scope)
        test_settings.start_pose = np.clip(test_settings.start_pose, 0, test_settings.start_scope - 1)

        if test_settings.start_pose in [39,40,41,42,43,48,51,68,79]:
            test_settings.start_pose = 1

        test_settings.start_pose_x = vector_info[1]
        test_settings.start_pose_y = vector_info[2]
        test_settings.start_pose_yaw = vector_info[3] * test_settings.yaw_scope

        test_settings.target_pose = int(((vector_info[4] -  test_settings.min) /  (test_settings.max -  test_settings.min)) * test_settings.target_scope)
        test_settings.target_pose = np.clip(test_settings.target_pose, 0, test_settings.target_scope - 1)

        test_settings.weather = int(((vector_info[5] -  test_settings.min) /  (test_settings.max -  test_settings.min)) * test_settings.weather_scope)
        
        test_settings.vehicles = []
        for i in range(test_settings.target_scope - 1):
            v_x = vector_info[6 + i * 2]
            v_y = vector_info[7 + i * 2] 
            test_settings.vehicles.append((v_x, v_y))
        return test_settings