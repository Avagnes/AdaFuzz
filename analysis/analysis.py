import json
from pathlib import Path
import numpy as np
import pickle
import os


def analysis(paths):
    print("path number: ", len(paths))
    for path in paths:
        path_valid = Path(path)
        with open(path_valid, 'rb') as f:
            all_seeds, results, result_time = pickle.load(f)
        print(path_valid.stem, len(all_seeds), len(results))


def create_json():
    paths = {}
    envs = ['ACAS_Xu', 'IL_CARLA', 'MARL_CoopNavi', 'RL_BipedalWalker', 'RL_CARLA']
    algos = ['MDPFuzz', 'G-Model', 'CureFuzz', 'AdaFuzz']
    for env in envs:
        paths[env] = {}
        for algo in algos:
            paths[env][algo] = [""]
    root = Path(__file__)
    with open(root.parent / "data_path.json", 'w') as f:
        json.dump(paths, f, indent=2)



def cal_area():
    from scipy import integrate
    envs = ['acas', 'marl', 'RL_CARLA', 'IL_CARLA', 'walker']
    root = Path(__file__).parent
    for env in envs:
        print(env)
        for file in (root / 'total' / env).glob('*.pickle'):
            with open(file, 'rb') as f:
                data = pickle.load(f)
            x, y = zip(*data)
            x = list(x)
            y = list(y)
            area = integrate.trapezoid(y, x)
            print('\t', file.stem, y[0], area/0.05/y[0], area/0.05)


if __name__ == "__main__":
    create_json()
    # cal_area()
    # root = Path(__file__)
    # with open(root.parent / "data_path.json", 'r') as f:
    #     paths = json.load(f)
    # for env in paths.keys():
    #     print(env)
    #     for algo in paths[env].keys():
    #         print(algo)
    #         analysis(paths[env][algo])
