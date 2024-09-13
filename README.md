## AdaFuzz

Here we give the execution command of AdaFuzz of different environments. For installing proper python environment, please follow the instructions of `README.md` in different environment folders.

#### ACAS_Xu
```bash
cd ACAS_Xu
python adafuzz.py
```

#### MARL_CoopNavi
```bash
cd MARL_CoopNavi/maddpg/experiments
python adafuzz.py
```

#### RL_BipedalWalker
```bash
cd RL_BipedalWalker
python adafuzz.py
```

#### RL_CARLA
```bash
cd RL_CARLA
python benchmark_agent.py --algo adafuzz
```

#### IL_CARLA
```bash
cd IL_CARLA/carla_lbc
python benchmark_agent.py --algo adafuzz
```

For environments **RL_CARLA** and **IL_CARLA**, remember to start the CARLA simulator first. 

For data analysis, just copy the generated pickle file path to `./analysis/data_path.json` and then run the python file in `./analysis` of its corresponding environment.