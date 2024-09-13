import numpy as np
import pickle
import os
import json
import tqdm
import time
import argparse


state_range = [
    (10, 1100),  # acas_speed
    (-60261, 60261),  # x2
    (-60261, 60261),  # y2
    (-np.pi, np.pi)  # auto_theta
]


class IntData:
    def __init__(self, algo):
        self.algo = algo
        with open('ablation_path.json', 'r') as f:
            file_paths = json.load(f)['ACAS_Xu']
        file_paths = file_paths[algo]
        self.dis_map_list = []
        max_distances = [
            max_val - min_val
            for min_val, max_val in state_range
        ]
        max_distance = np.sqrt(sum(d**2 for d in max_distances))
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                all_seeds, result, result_time = pickle.load(f)
            standardized_states = np.array(all_seeds, dtype=np.float32)  #
            LEN = len(standardized_states)
            dis_map = np.zeros((LEN, LEN), dtype=np.float32)
            for i in tqdm.tqdm(range(LEN)):
                dis_map[i,i] = 0
                dis_map[i,i+1:] = np.linalg.norm(standardized_states[i+1:] - standardized_states[i], axis=1)/max_distance
                dis_map[i+1:,i] = dis_map[i,i+1:]
            # del standardized_states
            self.dis_map_list.append(dis_map)

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
        return count

def search(adjacent_matrix):
    if adjacent_matrix.all():
        return 1
    LEN = len(adjacent_matrix)
    adjacent_matrix[np.eye(len(adjacent_matrix), dtype=np.bool_)] = False
    p = np.ones(LEN, dtype=np.float32)/2
    p_nest = np.ones(LEN, dtype=np.float32)/2
    random_data = np.zeros(LEN, dtype=np.float32)
    marks = np.zeros(LEN, dtype=np.bool_)
    count = 0
    del_nodes = np.zeros(len(p), dtype=np.bool_)
    adjacent_matrix_desk = np.copy(adjacent_matrix)
    while len(adjacent_matrix):
        random_data[:] = np.float32(np.random.uniform(size=len(p)))
        marks[:] = random_data < p
        for k in range(len(marks)):
            if del_nodes[k]:
                continue
            neighbors = adjacent_matrix[k,:] == True
            if not marks[neighbors].any():
                del_nodes = del_nodes | neighbors
                del_nodes[k] = True
                count = count + 1
                continue
            dt = p[neighbors].sum(dtype=np.float32)
            if dt < 2:
                p_nest[k] = 2*p[k]
            else:
                p_nest[k] = p[k] / 2
        del_nodes[:] = ~del_nodes
        N = del_nodes.sum()
        if not del_nodes.any():
            break
        adjacent_matrix_desk[:N,:] = adjacent_matrix[del_nodes,:]
        adjacent_matrix_desk[:,:N] = adjacent_matrix[:,del_nodes]
        adjacent_matrix_desk = adjacent_matrix_desk[:N,:N]
        adjacent_matrix = adjacent_matrix[:N,:N]
        adjacent_matrix[:,:] = adjacent_matrix_desk[:,:]
        random_data = random_data[:N]
        marks = marks[:N]
        p[:N] = p_nest[del_nodes].clip(max=1/2)
        p = p[:N]
        p_nest = p_nest[:N]
        del_nodes = del_nodes[:N]
        del_nodes[:] = False
    return count


for algo in ['MDPFuzz', 'G-Model', 'CureFuzz', 'AdaFuzz']:
    searcher = IntData(algo)
    data = []
    for eta in np.linspace(0, 0.05, 40):
        ans = searcher(eta)
        data.append((eta, ans))
        with open('./total/acas/'+algo+'.pickle', 'wb') as f:
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
    #         with open('./total/acas/'+algo+'.pickle', 'wb') as f:
    #             pickle.dump(data, f)






