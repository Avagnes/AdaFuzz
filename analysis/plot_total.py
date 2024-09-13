import matplotlib
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


envs = ['acas', 'marl', 'RL_CARLA', 'IL_CARLA', 'walker']
env_names = {'acas': 'ACAS_Xu', 'marl': 'MARL_CoopNavi', 'walker': 'RL_BipedalWalker', 'RL_CARLA': 'RL_CARLA', 'IL_CARLA': 'IL_CARLA'}
root = Path(__file__).parent
fig = plt.figure(figsize=(20.4,4.08))
for k, env in enumerate(envs):
    fig.add_axes([0.195*k+0.05, 0.15, 0.16, 0.75])
    algos = []
    for file in (root / 'total' / env).glob('*.pickle'):
        algos.append(file.stem)
        with open(file, 'rb') as f:
            data = pickle.load(f)
        x, y = zip(*data)
        x = list(x)
        y = list(y)
        data = None
        plt.plot(x, y)
    plt.title(env_names[env])
    plt.xlim([0, 0.05])
    plt.grid()
    legend = plt.legend(algos, loc=1)
ax = fig.add_axes([0.02, 0.08, 0.95, 0.75], frameon=False)
ax.set_xticks([])
ax.set_yticks([])
plt.xlabel('Threshold $\\eta$')
plt.ylabel('Different Test Case Number $f(\\mathbb{S},\\eta)$')
plt.savefig(root / 'total' / 'result.pdf', transparent=True)