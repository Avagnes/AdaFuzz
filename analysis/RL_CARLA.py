import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import scipy.integrate
import tqdm
import torch
import scipy
import time


class IntData:
    def __init__(self, algo) -> None:
        self.algo = algo
        self.device = torch.device('cuda')
        with open('data_path.json', 'r') as f:
            file_paths = json.load(f)
        file_paths = file_paths['RL_CARLA'][algo]
        self.dis_map_list = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                all_seeds, result, result_time = pickle.load(f)
                LEN = (np.array(result_time)<=2).sum()
                standardized_states = [np.array([float(s_k) for s_k in s]) for s in result[:LEN]]
                standardized_states.sort(key=lambda x: len(x))
                LEN = len(standardized_states)
                dis_map = np.zeros((LEN, LEN))
                for i in tqdm.tqdm(range(LEN)):
                    dis_map[i,i] = 0
                    for j in range(i+1,LEN):
                        if standardized_states[i][-1] != standardized_states[j][-1]:
                            dis_map[i,j] = 1
                            continue
                        l1 = len(standardized_states[i])-3
                        t1 = standardized_states[j][:l1] - standardized_states[i][:-3]
                        t2 = standardized_states[j][-3:] - standardized_states[i][-3:]
                        if t2[-1] != 0:
                            t2[-1] = 100
                        max_distance = np.sqrt(40804*4+72900+10000+40804*(len(standardized_states[i])-6))
                        dis_map[i,j] = (np.linalg.norm(np.concatenate([t1,t2]))/max_distance)
                    dis_map[i+1:,i] = dis_map[i,i+1:]
                del standardized_states
                self.dis_map_list.append(torch.tensor(dis_map).to(torch.float32).to(self.device))
        self.data = []

    def __call__(self, threshold, times=1):
        print('search threshold ', threshold)
        count = []
        for dis_map in self.dis_map_list:
            tmp = []
            for _ in range(times):
                search_map = dis_map <= threshold
                tmp.append(search(search_map))
            count.append(max(tmp))
        print(count)
        count = np.average(count)
        self.data.append((threshold, count))
        return count

def search(adjacent_matrix):
    if adjacent_matrix.all():
        return 1
    LEN = len(adjacent_matrix)
    adjacent_matrix[torch.eye(len(adjacent_matrix), dtype=torch.bool)] = False
    device = torch.device('cuda')
    p = torch.ones(LEN, dtype=torch.float32, device=device)/2
    p_nest = torch.ones(LEN, dtype=torch.float32, device=device)/2
    random_data = torch.zeros(LEN, dtype=torch.float32, device=device)
    marks = torch.zeros(LEN, dtype=torch.bool, device=device)
    count = 0
    while len(adjacent_matrix):
        random_data[:] = torch.rand(len(p), dtype=torch.float32, device=device)
        marks[:] = random_data < p
        del_nodes = torch.zeros(len(p), dtype=torch.bool, device=device)
        for k in range(len(marks)):
            if del_nodes[k]:
                continue
            neighbors = adjacent_matrix[k,:] == True
            if not marks[neighbors].any():
                del_nodes = del_nodes | neighbors
                del_nodes[k] = True
                count = count + 1
                continue
            dt = p[neighbors].sum(dtype=torch.float32)
            if dt < 2:
                p_nest[k] = 2*p[k]
            else:
                p_nest[k] = p[k] / 2
        adjacent_matrix = adjacent_matrix[~del_nodes,:][:,~del_nodes]
        random_data = random_data[~del_nodes]
        marks = marks[~del_nodes]
        tmp = p
        p = p_nest
        p_nest = tmp[~del_nodes]
        p = p[~del_nodes].clip(max=1/2)
    return count


for algo in ['MDPFuzz', 'G-Model', 'CureFuzz', 'AdaFuzz']:
    searcher = IntData(algo)
    data = []
    for eta in np.linspace(0, 0.05, 40):
        ans = searcher(eta)
        data.append((eta, ans))
        with open('./total/RL_CARLA/'+algo+'.pickle', 'wb') as f:
            pickle.dump(data, f)


# If you want adaptive searching, uncomment the following
    # data = []
    # ans = searcher(0)
    # data.append([0,ans])
    # ans = searcher(0.05,20)
    # data.append([0.05,ans])
    # while len(data)<40:
    #     # print(np.array(data))
    #     tau = np.random.uniform(0,0.05)
    #     ans = searcher(tau)
    #     for j in range(1,len(data)):
    #         if data[j][0] > tau:
    #             break
    #     if ans > data[j][1]:
    #         data.insert(j, (tau, ans))
    #         print('insert!', len(data))
    #         data = np.array(data)
    #         select = np.zeros(len(data), dtype=np.bool_)
    #         select[0] = True
    #         select[-1] = True
    #         i=1
    #         j=len(data)-1
    #         while i<j:
    #             idx = data[i:j+1,1].argmax()
    #             select[idx+i] = True
    #             i = idx+i+1
    #         data = data[select,:].tolist()
    #         with open('./total/RL_CARLA/'+algo+'.pickle', 'wb') as f:
    #             pickle.dump(data, f)



