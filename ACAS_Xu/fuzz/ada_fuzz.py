import numpy as np
import copy
import torch.nn as nn
import torch
import torch.optim as optim
import math
import tqdm

class RND(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=64):
        super(RND, self).__init__()
        self.target_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.predictor_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
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
        def __init__(self, seed, entropy, reward):
            self.corpus = [seed]
            self.entropy = [entropy]
            self.rewards = [reward]
            self.childrenIndex = [[]]
            self.parentIndex = [0]
            self.current_index = 0

        def get_seed(self):
            choose_index = \
            np.random.choice(range(len(self.corpus)), 1, p=np.array(self.entropy) / np.sum(self.entropy))[0]
            self.current_index = choose_index
            return self.corpus[choose_index], self.rewards[choose_index]

        def add_seed(self, seed, entropy, reward):
            self.corpus.append(copy.deepcopy(seed))
            self.entropy.append(copy.deepcopy(entropy))
            self.childrenIndex.append([])
            self.parentIndex.append(self.current_index)
            self.rewards.append(reward)
            self.update_entropy(self.current_index, len(self.entropy) - 1)

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
            return np.max(self.entropy)
        
    def __init__(self, family_size, corpus_factor=1, corpus_ratio=10):
        self.family_size = family_size
        self.corpus_ratio = corpus_ratio
        self.corpus_factor = corpus_factor
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
        self.optimizer = optim.Adam(list(self.rnd.predictor_net.parameters()), lr=1e-4)

    def add_family(self, seed, entropy, reward):
        self.families.append(self.Family(seed, entropy, reward))
        self.samplePr.append(entropy)
        self.seed_number += 1
        self.family_number += 1

    def get_seed(self):
        # new_prob = np.array(self.intrinsic_reward) + np.exp(-np.round(self.rewards)*0.1) + np.array(self.entropy)
        choose_index = np.random.choice(range(len(self.families)), 1, p=np.array(self.samplePr) / np.sum(self.samplePr))[0]
        self.current_family = choose_index
        return self.families[choose_index].get_seed()

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
        self.threshold = seed_sum / seed_len * (1 + self.corpus_factor * np.log(tmp))

    def add_seed(self, seed, entropy, reward):
        self.families[self.current_family].add_seed(seed, entropy, reward)
        self.samplePr[self.current_family] = self.families[self.current_family].represent_entropy()
        self.seed_number += 1

    def drop_current(self):
        choose_index = self.current_index
        if self.current_index != None:
            self.corpus.pop(choose_index)
            self.final_state.pop(choose_index)
            self.rewards.pop(choose_index)
            self.entropy.pop(choose_index)
            self.intrinsic_reward.pop(choose_index)
            self.original.pop(choose_index)
            self.count.pop(choose_index)
            self.current_index = None
    
    def compute_intrinsic_reward(self, states):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(states).to(self.device)
            target_out, predictor_out = self.rnd(state_tensor)
            intrinsic_reward = torch.pow(target_out - predictor_out, 2).sum()
        intrinsic_reward = intrinsic_reward.cpu().numpy()
        return intrinsic_reward
    
    
    def train_rnd(self, states, intrinsic_reward_scale=1.0, l2_reg_coeff=1e-4):
        state_tensor = torch.FloatTensor(states).to(self.device)
        target_out, predictor_out = self.rnd(state_tensor)
        intrinsic_reward = torch.pow(target_out[0,:] - predictor_out[0,:] , 2).sum(dim=0, keepdim=True)                
         # Compute the mean squared error between the predictor and target network outputs
        mse_loss = nn.MSELoss()(predictor_out, target_out)

        # Add L2 regularization to the loss
        l2_reg = 0
        for param in self.rnd.predictor_net.parameters():
            l2_reg += torch.norm(param)
        loss = mse_loss + l2_reg_coeff * l2_reg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        intrinsic_reward = intrinsic_reward[0].cpu().detach().numpy()
        if np.random.rand() < 0.1 or self.rnd_standard is None:
            self.rnd_standard = intrinsic_reward
    
        return intrinsic_reward


class SeedSpace:
    def __init__(self, dim=15, low=1, high=4):
        # seed consists of a vector between low (inclusive) and high (exclusive) with dimension dim
        self.low = low
        self.high = high
        self.dim = dim
        self.mutation_number = 3
        self.mutation_times = 0

    def random_generate(self):
        acas_speed = np.random.uniform(low=10, high=1100)
        row = np.random.uniform(low=1000, high=60261)
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x2 = row * np.cos(theta)
        y2 = row * np.sin(theta)
        bound1 = np.arcsin(-y2/np.linalg.norm([x2, y2]))
        if x2 > 0:
            bound1 = np.pi - bound1
        if bound1 > np.pi / 2:
            bound1, bound2 = np.pi/2, bound1
        else:
            bound1, bound2 =  bound1, np.pi/2
        auto_theta = np.random.uniform(low=bound1, high=bound2)
        return (acas_speed, x2, y2, auto_theta)

    def mutate(self, seed):
        orig_acas_speed, orig_x2, orig_y2, orig_auto_theta = seed
        new_acas_speed = orig_acas_speed + np.random.uniform(-5, 5)
        new_x2 = orig_x2 + np.random.uniform(-5, 5)
        new_y2 = orig_y2 + np.random.uniform(-5, 5)
        new_auto_theta = orig_auto_theta + np.random.uniform(-0.2, 0.2)
        return (new_acas_speed, new_x2, new_y2, new_auto_theta)

    def create_data(div=15):
        data = np.random.uniform(0, 1, size=(80000,4)).tolist()
        print("Begin adaptive sort!")
        normal_cases = SeedSpace._adaptive_sort(data)
        return normal_cases

    def _adaptive_sort(candidates: list):  # a faster gpu version
        # np.random.seed()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16
        first_idx = np.random.randint(len(candidates))

        LENGTH = len(candidates[0])
        MID1 = int(len(candidates)/3)
        MID2 = int(len(candidates)/2)
        MID3 = MID1 * 2
        chosen_desk = torch.zeros(len(candidates), LENGTH, dtype=dtype, device=device)
        chosen_desk[0,:] = torch.tensor(np.array([candidates.pop(first_idx)]))
        chosen = chosen_desk[:1,:]
        for_chosen = torch.tensor(np.array(candidates), dtype=dtype, device=device)
        for_chosen_desk = torch.zeros(for_chosen.shape, dtype=dtype, device=device)
        minus_desk = torch.zeros((len(candidates), LENGTH), dtype=dtype, device=device)
        square_desk = torch.zeros((len(candidates), LENGTH), dtype=dtype, device=device)
        value_desk = torch.zeros((len(candidates), MID1), dtype=dtype, device=device)
        value_desk_desk = torch.zeros((len(candidates), MID1), dtype=dtype, device=device)
        
        pbar = tqdm.tqdm(total=len(for_chosen))
        for i in range(len(candidates)-1):
            if i == MID1:
                del value_desk_desk
                value_desk_desk = torch.zeros((value_desk.shape[0], MID2), dtype=dtype, device=device)
                value_desk_desk[:, :value_desk.shape[1]] = value_desk
                del value_desk
                value_desk = torch.clone(value_desk_desk)
            elif i == MID2:
                del value_desk_desk
                value_desk_desk = torch.zeros((value_desk.shape[0], MID3), dtype=dtype, device=device)
                value_desk_desk[:, :value_desk.shape[1]] = value_desk
                del value_desk
                value_desk = torch.clone(value_desk_desk)
            elif i == MID3:
                del value_desk_desk
                value_desk_desk = torch.zeros((value_desk.shape[0], len(candidates)), dtype=dtype, device=device)
                value_desk_desk[:, :value_desk.shape[1]] = value_desk
                del value_desk
                value_desk = torch.clone(value_desk_desk)
            minus_desk[:,:] = for_chosen - chosen[-1,:]
            value_desk[:,i] = torch.square(minus_desk, out=square_desk).sum(axis=1)
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
            square_desk = square_desk[:-1,:]
            pbar.update(1)

        pbar.close()
        return chosen.cpu().numpy().tolist()

    def map_normal_case(normal_case):
        acas_speed, row, theta, auto_theta = normal_case 
        acas_speed     = acas_speed * (1100-10) + 10
        row            = row * (60261-1000) + 1000
        theta          = theta * 2 * np.pi - np.pi
        x2             = row * np.cos(theta)
        y2             = row * np.sin(theta)
        _theta = -np.arcsin(np.sin(theta))
        if x2 > 0:
            _theta = np.pi - _theta
        if _theta > np.pi / 2:
            bound1, bound2 = np.pi / 2, _theta
        else:
            bound1, bound2 = _theta, np.pi / 2
        auto_theta     = auto_theta * (bound2-bound1) + bound1
        return acas_speed, x2, y2, auto_theta
