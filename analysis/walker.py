import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import tqdm
import torch
import time
import scipy
import math
import argparse


class IntData:
    def __init__(self, algo) -> None:
        self.algo = algo
        self.device = torch.device('cuda')
        with open('data_path.json', 'r') as f:
            file_paths = json.load(f)['RL_BipedalWalker']
        file_paths = file_paths[algo]
        self.dis_map_list = []
        self.count = []
        for file_path in file_paths:
            seed_set = set()
            with open(file_path, 'rb') as f:
                all_seeds, result, result_time = pickle.load(f)
            standardized_states = np.array(all_seeds).tolist()
            for seed in standardized_states:
                seed_set.add(tuple(seed))
            self.count.append(len(seed_set))
        self.count = np.average(self.count)
        self.data = []

    def __call__(self, threshold):
        self.data.append((threshold, self.count))
        return self.count
    
    def save(self):
        with open('./total/walker/'+self.algo+'.pickle', 'wb') as f:
            pickle.dump(self.data, f)
        print(self.algo, ' data saved!')


algos = ['MDPFuzz', 'G-Model', 'CureFuzz','AdaFuzz']
for algo in algos:
    data = IntData(algo)
    for k in np.linspace(0, 0.05, 40):
        data(k)
    data.save()