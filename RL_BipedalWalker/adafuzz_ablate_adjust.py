import argparse, importlib, os, sys, time, copy, tqdm, pickle, gym, yaml
import numpy as np
import torch as th
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecVideoRecorder
import utils.import_envs
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.exp_manager import ExperimentManager
from utils.utils import StoreDict
from fuzz.ada_fuzz import AdaFuzz, SeedSpace
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="BipedalWalkerHardcore-v3")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
    parser.add_argument("--algo", help="RL Algorithm", default="tqc", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument("--hours", help="time limit", default=1, type=float)
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    is_atari = ExperimentManager.is_atari(env_id)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=not args.no_render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        kwargs.update(dict(buffer_size=1))

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

    # np.random.seed(2021)

    '''
    video_length = args.n_timesteps
    env = VecVideoRecorder(
    env,
    "./recording/test",
    record_video_trigger=lambda x: x == 0,
    video_length=video_length,
    name_prefix=f"{algo}-{env_id}",
    )
    obs = env.reset(states)
    '''

    # Deterministic by default except for atari games
    stochastic = args.stochastic or is_atari and not args.deterministic
    deterministic = not stochastic

    episode_rewards, episode_lengths = [], []
    ep_len = 0
    successes = []
    fuzzer = AdaFuzz(10)

    # HACK: start fuzzing
    start_fuzz_time = time.time()
    seed_space = SeedSpace()
    result = []
    result_time = []
    all_seeds = []
    state = None
    pbar = tqdm.tqdm(total=int(args.hours * 3600 * 8))

    while time.time()-start_fuzz_time < 3600*args.hours:
        while len(fuzzer.families) < 10:
            seed = seed_space.random_generate()
            all_seeds.append(tuple(seed))
            episode_reward = 0.0
            obs = env.reset(seed)
            sequences = [obs[0]]
            pbar.update(1)
            for _ in range(args.n_timesteps):
                action, state = model.predict(obs, state=state, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                sequences.append(obs[0])
                # if not args.no_render:
                #     env.render("human")
                episode_reward += reward[0]
                if done:
                    break
            if done or episode_reward < 10:
                result.append(tuple(seed))
                result_time.append(time.time())
                continue
            intrinsic_reward = fuzzer.train_rnd(sequences)
            samplePr = np.exp(-episode_reward*0.1) + np.exp(intrinsic_reward*0.1)
            fuzzer.add_family(seed, samplePr, episode_reward)

        seed, reward = fuzzer.get_seed()
        mutate_seed = seed_space.mutate(seed)
        all_seeds.append(mutate_seed)
        episode_reward = 0.0
        obs = env.reset(seed)
        sequences = [obs[0]]
        for _ in range(args.n_timesteps):
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, infos = env.step(action)
            sequences.append(obs[0])
            # if not args.no_render:
            #     env.render("human")
            episode_reward += reward[0]
            if done:
                break
        if done or episode_reward < 10:
            result.append(tuple(mutate_seed))
            result_time.append((time.time()-start_fuzz_time)/3600)
            fuzzer.delete_family(fuzzer.current_family)
            continue
        else:
            intrinsic_reward = fuzzer.train_rnd(sequences)
            samplePr = np.exp(-episode_reward*0.1) + np.exp(intrinsic_reward*0.1)
            fuzzer.add_seed(mutate_seed, samplePr, episode_reward)
        pbar.update(1)
        pbar.set_postfix({'Found': len(result), 'Families': fuzzer.family_number, 'corpus': fuzzer.seed_number, 'threshold': fuzzer.threshold})
    pbar.close()

    with open("AdaFuzz"+datetime.now().strftime("%m_%d_%Y_%H_%M_%S"), "wb") as f:
        pickle.dump((all_seeds, result, result_time), f)


    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    if not args.no_render:
        if args.n_envs == 1 and "Bullet" not in env_id and not is_atari and isinstance(env, VecEnv):
            while isinstance(env, VecEnvWrapper):
                env = env.venv
            if isinstance(env, DummyVecEnv):
                env.envs[0].env.close()
            else:
                env.close()
        else:
            env.close()


if __name__ == "__main__":
    f = open('./results/fuzz.txt', 'w', buffering=1)
    sys.stdout = f
    main()
