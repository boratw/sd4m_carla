#!/usr/bin/env python

# Copyright (c) 2023 Taewoo Kim

import glob
import os
import sys
import time
import carla

import numpy as np

from carla import VehicleLightState as vls
from carla import TrafficLightState as tls
import tensorflow.compat.v1 as tf

from networks.sd4m_explorer import Skill_Explorer

import random
import pickle
from datetime import datetime


tf.disable_eager_execution()
sess = tf.Session()

vehicles_list = []
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

agent_num = 150

state_len = 11
action_len = 4
goal_len = 2
traj_len = 50
traj_track_len = 10
latent_len = 3
latent_body_len = 2
latent_preserve = 4
task_num = 3
env_num = 64

explorer_batch = 64

log_name = "train_log/Train2_sac/log_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file = open(log_name + ".txt", "wt")


history_policy =  [[] for _ in range(task_num)]

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

try:
    with sess.as_default():
        explorer = [Skill_Explorer(state_len,  action_len, goal_len, name=str(i)) for i in range(task_num)]
        explorer_savers = [tf.train.Saver(max_to_keep=0, var_list=explorer[i].trainable_dict) for i in range(task_num)]

        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(task_num):
            explorer[i].network_initialize()
        world = client.get_world()


        settings = world.get_settings()
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        #settings.no_rendering_mode = True
        world.apply_settings(settings)

        bp_library = world.get_blueprint_library()
        blueprints = [bp_library.find("vehicle.neubility.delivery"),
                    bp_library.find("vehicle.neubility.delivery_leftbig"),
                    bp_library.find("vehicle.neubility.delivery_rightbig")]
        for exp in range(1, 2001):
            cur_move = [0.] * task_num
            cur_reward = [0.] * task_num
            for task in range(task_num):
                vehicles_list = []
                for x in range(8):
                    for y in range(8):
                        i = random.randrange(3)                                                                                                                                                                                                                                                                      
                        spawn_point = carla.Transform(carla.Location(x * 200. - 800., y * 200. - 800., 3.0), carla.Rotation(0, 0, 0))
                        actor = world.spawn_actor(blueprints[task], spawn_point)
                        vehicles_list.append(actor)

                        
                for step in range(30):
                    world.tick()

                states = []
                survive_vector = [True for _ in vehicles_list]
                first_traj_pos = []
                first_traj_yaw = []
                for actor in vehicles_list:
                    v = actor.get_velocity()
                    a = actor.get_acceleration()
                    tr = actor.get_transform()
                    yaw = tr.rotation.yaw * -0.017453293
                    yawsin = np.sin(yaw)
                    yawcos = np.cos(yaw)
                    vx, vy = rotate(v.x, v.y, yawsin, yawcos)
                    ax, ay = rotate(a.x, a.y, yawsin, yawcos)
                    states.append([0., 0., 0., vx, vy, ax, ay, 1., 0., tr.rotation.roll, tr.rotation.pitch])
                for step in range(500):
                    
                    if step % traj_len == 0:
                        if (step // traj_len) % latent_preserve == 0:
                            goal_d = np.random.uniform(-1.047197551, 1.047197551, (env_num,))
                            current_goal = np.array([[np.cos(d) * 0.2, np.sin(d) * 0.2] for d in goal_d])
                        state_traj = [[] for _ in range(env_num)]
                        body_traj = [[] for _ in range(env_num)]
                        goal_traj = [[] for _ in range(env_num)]
                        action_traj = [[] for _ in range(env_num)]

                        first_traj_pos = []
                        first_traj_yaw = []
                        for actor in vehicles_list:
                            tr = actor.get_transform()
                            first_traj_pos.append([tr.location.x, tr.location.y])
                            first_traj_yaw.append([np.sin(tr.rotation.yaw * -0.017453293), np.cos(tr.rotation.yaw * -0.017453293)])  

                    actions = explorer[task].get_action(states, current_goal, False)

                    vehiclecontrols = []
                    for i, actor in enumerate(vehicles_list):

                        dropout = random.randrange(40)
                        if dropout < action_len:
                            actions[i][dropout] = np.tanh(actions[i][dropout] + np.random.normal(0., 0.5))

                        v = actor.get_velocity()
                        vs = np.sqrt(v.x * v.x + v.y * v.y)
                        control = carla.VehicleControl()
                        control.fl=float(actions[i][0] * (300.0 if task == 1 else 200.0))
                        control.fr=float(actions[i][1] * (300.0 if task == 2 else 200.0))
                        control.bl=float(actions[i][2] * (300.0 if task == 1 else 200.0))
                        control.br=float(actions[i][3] * (300.0 if task == 2 else 200.0))
                        control.gear = 1
                        control.manual_gear_shift = True
                        control.hand_brake = False
                        control.reverse = False
                        
                        vehiclecontrols.append(carla.command.ApplyVehicleControl(actor, control))

                    client.apply_batch(vehiclecontrols)
                    world.tick()

                    nextstates = []
                    for i, actor in enumerate(vehicles_list):
                        tr = actor.get_transform()
                        v = actor.get_velocity()
                        a = actor.get_acceleration()
                        tr = actor.get_transform()
                        yaw = tr.rotation.yaw * -0.017453293
                        f = tr.get_forward_vector()
                        px, py = rotate(tr.location.x - first_traj_pos[i][0], tr.location.y - first_traj_pos[i][1], first_traj_yaw[i][0], first_traj_yaw[i][1])
                        vx, vy = rotate(v.x, v.y, first_traj_yaw[i][0], first_traj_yaw[i][1])
                        ax, ay = rotate(a.x, a.y, first_traj_yaw[i][0], first_traj_yaw[i][1])
                        fx, fy = rotate(f.x, f.y, first_traj_yaw[i][0], first_traj_yaw[i][1])
                        nextstates.append([float(step % traj_len), px, py, vx, vy, ax, ay, fx, fy, tr.rotation.roll, tr.rotation.pitch])


                        score = 1. - np.sqrt((current_goal[i][0] - (px- states[i][1])) ** 2 + (current_goal[i][1] - (py - states[i][2])) ** 2) * 2.5
                        reward = 0.1 - (abs(actions[i][0] - actions[i][2]) + abs(actions[i][1] - actions[i][3])) * 0.1
                        survive = (abs(tr.rotation.roll) < 45) and (abs(tr.rotation.pitch) < 45)
                        if survive_vector[i] :
                            if survive:
                                history_policy[task].append([states[i], nextstates[i][:], actions[i], current_goal[i], [score + reward], [1.]])
                            else:
                                history_policy[task].append([states[i], nextstates[i][:], actions[i], current_goal[i], [score - 1.], [0.]])
                                survive_vector[i] = survive


                        cur_reward[task] += score
                        
                        state_traj[i].append(states[i])
                        action_traj[i].append(actions[i])
                        if step % traj_track_len == traj_track_len - 1:
                            body_traj[i].append(np.array([px, py]))
                        nextstates[-1][0] += 1.

                    states = nextstates
                    if step % traj_len == traj_len - 1:
                        for i, actor in enumerate(vehicles_list):
                            skill_moved = np.sqrt(body_traj[i][-1][0] ** 2 + body_traj[i][-1][1] ** 2)
                            cur_move[task] += skill_moved
             
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
                cur_move[task] /= env_num
                cur_reward[task] /= env_num
                print("Episode " + str(exp) + " Task " + str(task) +  " Move " +  str(cur_move[task] ) +  " Reward " +  str(cur_reward[task]) )

            for iter in range(16):
                for iter2 in range(32):
                    for task in range(task_num):
                        if len(history_policy[task]) > explorer_batch:
                            dic = random.sample(range(len(history_policy[task])), explorer_batch)

                            state_dic = [history_policy[task][x][0] for x in dic]
                            nextstate_dic = [history_policy[task][x][1] for x in dic]
                            action_dic = [history_policy[task][x][2] for x in dic]
                            goal_dic = [history_policy[task][x][3] for x in dic]
                            reward_dic = [history_policy[task][x][4] for x in dic]
                            survive_dic = [history_policy[task][x][5] for x in dic]

                            explorer[task].optimize(state_dic, nextstate_dic, action_dic, goal_dic, reward_dic, survive_dic, exp)
                for task in range(task_num):
                    explorer[task].network_intermediate_update()
                    
            for i in range(task_num):
                explorer[i].log_print()

            log_file.write(str(exp) + "\t")
            for i in range(task_num):
                log_file.write(str(cur_move[i])  + "\t" + str(cur_reward[i]) + "\t")
            for i in range(task_num):
                log_file.write(explorer[i].current_log())
            log_file.write("\n")
            log_file.flush()

            for i in range(task_num):
                explorer[i].network_update()

            if exp % 50 == 0:
                for i in range(task_num):
                    explorer_savers[i].save(sess, log_name + "_explorer" + str(i) + "_" + str(exp) + ".ckpt")

            for task in range(task_num):
                history_policy[task] = history_policy[task][(len(history_policy[task] ) // 32):]



finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)