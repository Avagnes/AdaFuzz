from pathlib import Path
import pandas as pd
import numpy as np
import tqdm
import time, copy, pickle
import bird_view.utils.bz_utils as bzu
import bird_view.utils.carla_utils as cu
from bird_view.models.common import crop_birdview
from fuzz.ada_fuzz import AdaFuzz, SeedSpace
from fuzz.replayer import replayer
import carla
from carla import ColorConverter
from carla import WeatherParameters
from datetime import datetime
import os
import traceback
import copy

def _paint(observations, control, diagnostic, debug, env, show=False):
    import cv2
    import numpy as np
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    CROP_SIZE = 192
    X = 176
    Y = 192 // 2
    R = 2
    birdview = cu.visualize_birdview(observations['birdview'])
    birdview = crop_birdview(birdview)

    if 'big_cam' in observations:
        canvas = np.uint8(observations['big_cam']).copy()
        rgb = np.uint8(observations['rgb']).copy()
    else:
        canvas = np.uint8(observations['rgb']).copy()

    def _stick_together(a, b, axis=1):
        if axis == 1:
            h = min(a.shape[0], b.shape[0])
            r1 = h / a.shape[0]
            r2 = h / b.shape[0]
            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))
            return np.concatenate([a, b], 1)
            
        else:
            h = min(a.shape[1], b.shape[1])
            r1 = h / a.shape[1]
            r2 = h / b.shape[1]
            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))
            return np.concatenate([a, b], 0)

    def _write(text, i, j, canvas=canvas, fontsize=0.4):
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 9) for x in range(9+1)]
        cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, fontsize, WHITE, 1)

    _command = {
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            }.get(observations['command'], '???')

    if 'big_cam' in observations:
        fontsize = 0.8
    else:
        fontsize = 0.4

    for x, y in debug.get('locations', []):
        x = int(X - x / 2.0 * CROP_SIZE)
        y = int(Y + y / 2.0 * CROP_SIZE)
        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED

    for x, y in debug.get('locations_world', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)
        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED
    
    for x, y in debug.get('locations_birdview', []):
        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED       
 
    for x, y in debug.get('locations_pixel', []):
        S = R // 2
        if 'big_cam' in observations:
            rgb[y-S:y+S+1,x-S:x+S+1] = RED
        else:
            canvas[y-S:y+S+1,x-S:x+S+1] = RED
    for x, y in debug.get('curve', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)
        try:
            birdview[x,y] = [155, 0, 155]
        except:
            pass
    if 'target' in debug:
        x, y = debug['target'][:2]
        x = int(X - x * 4)
        y = int(Y + y * 4)
        birdview[x-R:x+R+1,y-R:y+R+1] = [0, 155, 155]
    ox, oy = observations['orientation']
    rot = np.array([
        [ox, oy],
        [-oy, ox]])
    u = observations['node'] - observations['position'][:2]
    v = observations['next'] - observations['position'][:2]
    u = rot.dot(u)
    x, y = u
    x = int(X - x * 4)
    y = int(Y + y * 4)
    v = rot.dot(v)
    x, y = v
    x = int(X - x * 4)
    y = int(Y + y * 4)

    if 'big_cam' in observations:
        _write('Network input/output', 1, 0, canvas=rgb)
        _write('Projected output', 1, 0, canvas=birdview)
        full = _stick_together(rgb, birdview)
    else:
        full = _stick_together(canvas, birdview)
    if 'image' in debug:
        full = _stick_together(full, cu.visualize_predicted_birdview(debug['image'], 0.01))
    if 'big_cam' in observations:
        full = _stick_together(canvas, full, axis=0)
    full = canvas

    if show:
        bzu.show_image('canvas', full)
    bzu.add_to_video(full)

def calculate_reward(prev_distance, cur_distance, cur_collid, cur_invade, cur_speed, prev_speed):
    reward = 0.0
    reward += np.clip(prev_distance - cur_distance, -10.0, 10.0)
    cur_speed_norm = np.linalg.norm(cur_speed)
    prev_speed_norm = np.linalg.norm(prev_speed)
    reward += 0.2 * (cur_speed_norm - prev_speed_norm)
    if cur_collid:
        reward -= 100 * cur_speed_norm
    if cur_invade:
        reward -= cur_speed_norm
    return reward

def save_pickle(replayer, log):
    corpus = []
    total_crash = len(replayer.corpus)
    for i in range(total_crash):
        single_crash = []
        temp_trans = replayer.corpus[i][0]
        single_crash.append([temp_trans.location.x, temp_trans.location.y, temp_trans.location.z, temp_trans.rotation.pitch, temp_trans.rotation.yaw, temp_trans.rotation.roll])
        temp_vehicleinfo = replayer.corpus[i][1]
        total_vehicle = len(temp_vehicleinfo)
        vehicle_info_crash = []
        for j in range(total_vehicle):
            temp_blue_print = temp_vehicleinfo[j][0]
            temp_trans = temp_vehicleinfo[j][1]
            temp_color = temp_vehicleinfo[j][2]
            temp_vehicle_id = temp_vehicleinfo[j][3]
            vehicle_info_crash.append([temp_blue_print.id, temp_blue_print.tags, temp_trans.location.x, temp_trans.location.y, temp_trans.location.z, temp_trans.rotation.pitch, temp_trans.rotation.yaw, temp_trans.rotation.roll, temp_color, temp_vehicle_id])
        corpus.append([single_crash, vehicle_info_crash])
        replayer.envsetting[i][2] = replayer.envsetting[i][3]

    replayer.corpus = corpus
    replayer.original = []
    with open('./results'+log+'/crash.pkl', 'wb') as handle:
        pickle.dump(replayer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Pickle Saved!!!')

def load_pickle(pickle_path, prints):
    with open(pickle_path, 'rb') as handle:
        replayer = pickle.load(handle)
    corpus = []
    envsetting = []
    total_crash = len(replayer.corpus)

    for i in range(total_crash):
        temp_trans = replayer.corpus[i][0][0]
        single_trans = carla.Transform(carla.Location(x=temp_trans[0], y=temp_trans[1], z=temp_trans[2]), carla.Rotation(pitch=temp_trans[3], yaw=temp_trans[4], roll=temp_trans[5]))
        vehicle_info_crash = replayer.corpus[i][1]
        total_vehicle = len(vehicle_info_crash)
        singel_vehicle = []
        for j in range(total_vehicle):
            blue_print = prints.filter(vehicle_info_crash[j][0])[0]
            assert blue_print.tags == vehicle_info_crash[j][1]
            blue_print.set_attribute("role_name", "autopilot")
            color = vehicle_info_crash[j][8]
            vehicle_id = vehicle_info_crash[j][9]
            if color != None:
                blue_print.set_attribute("color", color)
            if vehicle_id != None:
                blue_print.set_attribute("driver_id", vehicle_id)

            trans = carla.Transform(carla.Location(x=vehicle_info_crash[j][2], y=vehicle_info_crash[j][3], z=vehicle_info_crash[j][4]), carla.Rotation(pitch=vehicle_info_crash[j][5], yaw=vehicle_info_crash[j][6], roll=vehicle_info_crash[j][7]))
            singel_vehicle.append((blue_print, trans))
        corpus.append((single_trans, singel_vehicle))
        envsetting.append([replayer.envsetting[i][0], replayer.envsetting[i][1], cu.PRESET_WEATHERS[replayer.envsetting[i][2]], replayer.envsetting[i][3]])
    replayer.corpus = corpus
    replayer.envsetting = envsetting
    return replayer

def run_single(env, weather, start, target, agent_maker, seed, log, show=False, replay=False, guide=True, args=None):
    env.seed = seed
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
    print('replay = ', replay)
    tsne_data = []
    spawn_points = env._map.get_spawn_points()
    if replay == True:
        temp_count = 0
        prints = env._blueprints.filter("vehicle.*")
        replayer_load = load_pickle('./results/crash.pkl', prints)
        print('Load Done!')
        env.replay = True
        env.replayer = replayer_load
        print(env.replayer.rewards)
        env.replayer.replay_list = list(np.arange(len(env.replayer.corpus)))
        agent = agent_maker()
        while len(env.replayer.replay_list) > 0:
            temp_count += 1
            print('current id: ', env.replayer.replay_list[-1])
            video_name = '902_' + str(env.replayer.replay_list[-1])
            bzu.init_video(save_dir='./results/new-videos/', save_path=video_name)
            env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
            diagnostics = list()
            result = {
                "weather": weather,
                "start": start,
                "target": target,
                "success": None,
                "t": None,
                "total_lights_ran": None,
                "total_lights": None,
                "collided": None,
            }
            seq_entropy = 0

            first_reward_flag = True
            total_reward = 0
            sequence = []
            while env.tick():
                observations = env.get_observations()
                if first_reward_flag == False:
                    cur_invade = (prev_invaded_frame_number != env._invaded_frame_number)
                    cur_collid = (prev_collided_frame_number != env._collided_frame_number)
                if first_reward_flag:
                    first_reward_flag = False
                    prev_distance = env._local_planner.distance_to_goal
                    prev_speed = observations['velocity']
                    prev_invaded_frame_number = env._invaded_frame_number
                    prev_collided_frame_number = env._collided_frame_number
                    cur_invade = False
                    cur_collid = False
                    if env.invaded:
                        cur_invade = True
                    if env.collided:
                        cur_collid = True

                reward = calculate_reward(prev_distance, env._local_planner.distance_to_goal, cur_collid, cur_invade, observations['velocity'], prev_speed)
                total_reward += reward
                prev_distance = env._local_planner.distance_to_goal
                prev_speed = observations['velocity']
                prev_invaded_frame_number = env._invaded_frame_number
                prev_collided_frame_number = env._collided_frame_number

                control, current_entropy, _ = agent.run_step(observations)
                seq_entropy += current_entropy
                diagnostic = env.apply_control(control)

                _paint(observations, control, diagnostic, agent.debug, env, show=show)
                
                diagnostic.pop("viz_img")
                diagnostics.append(diagnostic)
                # HACK: T-SNE
                # tsne_data.append(copy.deepcopy(current_tsne))
                # if env.is_failure() or env.is_success() or env._tick > 200:
                if env._tick > 250:
                    result["success"] = env.is_success()
                    result["total_lights_ran"] = env.traffic_tracker.total_lights_ran
                    result["total_lights"] = env.traffic_tracker.total_lights
                    result["collided"] = env.collided
                    result["t"] = env._tick
                    break
            print(total_reward)
            # if temp_count > 1:
            #     break

        # HACK: T-SNE
        # tsne_data = np.array(tsne_data)
        # print(tsne_data.shape)
        # with open('./results/tsne_crash.pkl', 'wb') as tsne_file:
        #     np.save(tsne_file, tsne_data)
    else:
        env.replayer = replayer()
        fuzzer = AdaFuzz(10)
        env.fuzzer = fuzzer
        agent = agent_maker()
        normal_cases = SeedSpace.create_data()

        def add_family():
            nonlocal all_seeds, result, result_time, fuzzer, env, agent, pbar, start_fuzz_time  # , tasks
            while len(fuzzer.families) < fuzzer.family_size:
                try:
                    # print('generate! number ', len(fuzzer.families))
                    env.first_run = False
                    env.generative = True
                    if len(normal_cases):
                        test_settings = SeedSpace.map_normal_case(normal_cases.pop(0))
                    else:
                        test_settings = SeedSpace.random_generate()
                    env.test_settings = test_settings
                    initial_check = env.init(start=test_settings.start_pose, target=test_settings.target_pose, weather=test_settings.weather)
                    start = test_settings.start_pose
                    target = test_settings.target_pose
                    weather = test_settings.weather + 1
                    # print(((env._start_pose, env.init_vehicles, (start, target, cu.PRESET_WEATHERS[weather], weather))), initial_check)
                    if initial_check == False:
                        print("random failure")
                        continue
                    test_case = []
                    test_case = test_case + [env._start_pose.location.x, env._start_pose.location.y, env._start_pose.rotation.yaw]
                    for i in range(len(test_settings.vehicles)):
                        test_case = test_case + [spawn_points[i].location.x + test_settings.vehicles[i][0], spawn_points[i].location.y + test_settings.vehicles[i][1]]
                    test_case = test_case + [spawn_points[test_settings.target_pose].location.x, spawn_points[test_settings.target_pose].location.y]
                    test_case = test_case + [test_settings.weather]
                    test_case = [ float('%.2f' % elem) for elem in test_case]
                    all_seeds.append(tuple(test_case))

                    start_pose = env._start_pose
                    diagnostics = list()
                    seq_entropy = 0
                    first_reward_flag = True
                    total_reward = 0
                    sequence = []
                    while env.tick():
                        observations = env.get_observations()
                        if first_reward_flag == False:
                            cur_invade = (prev_invaded_frame_number != env._invaded_frame_number)
                            cur_collid = (prev_collided_frame_number != env._collided_frame_number)
                        if first_reward_flag:
                            first_reward_flag = False
                            prev_distance = env._local_planner.distance_to_goal
                            prev_speed = observations['velocity']
                            prev_invaded_frame_number = env._invaded_frame_number
                            prev_collided_frame_number = env._collided_frame_number
                            cur_invade = False
                            cur_collid = False
                            if env.invaded:
                                cur_invade = True
                            if env.collided:
                                cur_collid = True

                        reward = calculate_reward(prev_distance, env._local_planner.distance_to_goal, cur_collid, cur_invade, observations['velocity'], prev_speed)
                        total_reward += reward
                        prev_distance = env._local_planner.distance_to_goal
                        prev_speed = observations['velocity']
                        prev_invaded_frame_number = env._invaded_frame_number
                        prev_collided_frame_number = env._collided_frame_number

                        control, current_entropy, _ = agent.run_step(observations)

                        temp = copy.deepcopy(observations['node'])
                        temp = np.hstack((temp, copy.deepcopy(observations['orientation']), copy.deepcopy(observations['velocity']), copy.deepcopy(observations['acceleration']), copy.deepcopy(observations['position']), copy.deepcopy(np.array([observations['command']]))))
                        vehicle_index = np.nonzero(observations['vehicle'])
                        vehicle_obs = np.zeros(3)
                        vehicle_obs[0] = vehicle_index[0].mean()
                        vehicle_obs[1] = vehicle_index[1].mean()
                        vehicle_obs[2] = np.sum(observations['vehicle']) / 1e5
                        temp = np.hstack((temp, vehicle_obs))

                        seq_entropy += current_entropy
                        diagnostic = env.apply_control(control)
                        diagnostic.pop("viz_img")
                        diagnostics.append(diagnostic)
                        sequence.append(temp)

                        # HACK: T-SNE
                        # current_tsne = np.array(current_tsne).flatten()
                        # tsne_data.append(current_tsne)

                        if env.is_failure() or env.is_success() or env._tick > 100:
                            break
                    if env.is_failure() or env.collided:
                        results.append(tuple(test_case))
                        result_time.append((time.time() - start_fuzz_time)/3600)
                        continue
                    pbar.update(1)
                    intrinsic_reward = env.fuzzer.train_rnd(sequence)
                    samplePr = np.exp(-total_reward*0.01)+np.exp(intrinsic_reward*0.01)
                    fuzzer.add_family((start_pose, env.init_vehicles, (start, target, cu.PRESET_WEATHERS[weather], weather)), samplePr, total_reward)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    continue

        print('fuzzing start!')
        start_fuzz_time = time.time()
        all_seeds = []
        results = []
        result_time = []
        pbar = tqdm.tqdm(total=3600 * args.hour / 8)
        while time.time() - start_fuzz_time < 3600 * args.hour:
            add_family()
            env.first_run = False
            env.generative = False
            try:
                initial_check = env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])
            except Exception as e:
                print(e)
                traceback.print_exc()
                initial_check = False
            # print(((env._start_pose, env.init_vehicles, (start, target, cu.PRESET_WEATHERS[weather], weather))), initial_check)
            if initial_check == False:
                # print(env.fuzzer.current_index)
                print('Trigger initial collision!!!')
                env.fuzzer.delete_family(env.fuzzer.current_family)
                continue                    
            test_case = []
            test_case = test_case + [env.fuzzer.current_pose.location.x, env.fuzzer.current_pose.location.y, env.fuzzer.current_pose.rotation.yaw]
            for v in env.fuzzer.current_vehicle_info:
                test_case = test_case + [v[1].location.x, v[1].location.y]
            test_case = test_case + [spawn_points[env.fuzzer.current_envsetting[1]].location.x, spawn_points[env.fuzzer.current_envsetting[1]].location.y]
            test_case = test_case + [1]
            test_case = [ '%.2f' % elem for elem in test_case]
            all_seeds.append(tuple(test_case))

            diagnostics = list()
            result = {
                "weather": weather,
                "start": start,
                "target": target,
                "success": None,
                "t": None,
                "total_lights_ran": None,
                "total_lights": None,
                "collided": None,
            }
            seq_entropy = 0
            first_reward_flag = True
            total_reward = 0
            sequence = []
            while env.tick():
                observations = env.get_observations()
                if first_reward_flag == False:
                    cur_invade = (prev_invaded_frame_number != env._invaded_frame_number)
                    cur_collid = (prev_collided_frame_number != env._collided_frame_number)
                if first_reward_flag:
                    first_reward_flag = False
                    prev_distance = env._local_planner.distance_to_goal
                    prev_speed = observations['velocity']
                    prev_invaded_frame_number = env._invaded_frame_number
                    prev_collided_frame_number = env._collided_frame_number
                    cur_invade = False
                    cur_collid = False
                    if env.invaded:
                        cur_invade = True
                    if env.collided:
                        cur_collid = True
                reward = calculate_reward(prev_distance, env._local_planner.distance_to_goal, cur_collid, cur_invade, observations['velocity'], prev_speed)
                total_reward += reward
                prev_distance = env._local_planner.distance_to_goal
                prev_speed = observations['velocity']
                prev_invaded_frame_number = env._invaded_frame_number
                prev_collided_frame_number = env._collided_frame_number

                control, current_entropy, _ = agent.run_step(observations)
                temp = copy.deepcopy(observations['node'])
                temp = np.hstack((temp, copy.deepcopy(observations['orientation']), copy.deepcopy(observations['velocity']), copy.deepcopy(observations['acceleration']), copy.deepcopy(observations['position']), copy.deepcopy(np.array([observations['command']]))))
                vehicle_index = np.nonzero(observations['vehicle'])
                vehicle_obs = np.zeros(3)
                vehicle_obs[0] = vehicle_index[0].mean()
                vehicle_obs[1] = vehicle_index[1].mean()
                vehicle_obs[2] = np.sum(observations['vehicle']) / 1e5
                temp = np.hstack((temp, vehicle_obs))
                seq_entropy += current_entropy
                diagnostic = env.apply_control(control)
                diagnostic.pop("viz_img")
                diagnostics.append(diagnostic)

                sequence.append(copy.deepcopy(temp))


                if env.is_failure() or env.is_success() or env._tick > 100:
                    result["success"] = env.is_success()
                    result["total_lights_ran"] = env.traffic_tracker.total_lights_ran
                    result["total_lights"] = env.traffic_tracker.total_lights
                    result["collided"] = env.collided
                    result["t"] = env._tick
                    break
            intrinsic_reward = env.fuzzer.train_rnd(sequence)
            # new_entropy = np.linalg.norm(temp - env.fuzzer.current_final_state)+seq_entropy
            samplePr = np.exp(-total_reward*0.01)+np.exp(intrinsic_reward*0.01)
            fuzzer.add_seed((env.fuzzer.current_pose, env.fuzzer.current_vehicle_info, env.fuzzer.current_envsetting), samplePr, total_reward)
            
            if env.is_failure() or env.collided:
                env.replayer.store((env.fuzzer.current_pose, env.fuzzer.current_vehicle_info), rewards=total_reward, entropy=0, cvg=intrinsic_reward,original=None, further_envsetting=env.fuzzer.current_envsetting)
                fuzzer.delete_family(fuzzer.current_family)
                results.append(tuple(test_case))
                result_time.append((time.time() - start_fuzz_time)/3600)
            else:
                fuzzer.update_threshold()
                check = np.array(fuzzer.samplePr) < fuzzer.threshold
                while check.any():
                    for k in range(len(check)):
                        if check[k]:
                            fuzzer.delete_family(k)
                            break
                    check = np.array(fuzzer.samplePr) < fuzzer.threshold
            pbar.set_postfix({'Found': len(results), 'Families': fuzzer.family_number, 'corpus': fuzzer.seed_number, 'threshold': fuzzer.threshold})
            pbar.update(1)
        with open("AdaFuzz"+datetime.now().strftime("%m_%d_%Y_%H_%M_%S"), "wb") as f:
            pickle.dump((all_seeds, results, result_time), f)
        print('finish')
        save_pickle(env.replayer, log)
        os._exit(0)

    return result, diagnostics


def run_benchmark(agent_maker, env, benchmark_dir, seed, resume, log, max_run=5, show=False, replay=False, guide=True, args=None):
    """
    benchmark_dir must be an instance of pathlib.Path
    """
    summary_csv = benchmark_dir / 'summary.csv'
    diagnostics_dir = benchmark_dir / 'diagnostics'
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary = list()
    total = len(list(env.all_tasks))

    if summary_csv.exists() and resume:
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame()

    num_run = 0

    for weather, (start, target), run_name in env.all_tasks:
        if resume and len(summary) > 0 and ((summary['start'] == start) \
                       & (summary['target'] == target) \
                       & (summary['weather'] == weather)).any():
            print (weather, start, target)
            continue

        diagnostics_csv = str(diagnostics_dir / ('%s.csv' % run_name))
        result, diagnostics = run_single(env, weather, start, target, agent_maker, seed, log, show=show, replay=replay, guide=guide, args=args)

        summary = summary.append(result, ignore_index=True)
        pd.DataFrame(summary).to_csv(summary_csv, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)
        break
