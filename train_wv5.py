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

from networks.sd4m_learner import Skill_Learner
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

state_len = 10
action_len = 4
goal_len = 2
traj_len = 25
traj_track_len = 5
latent_len = 3
latent_body_len = 2
latent_preserve = 4
task_num = 3
env_num = 64

learner_batch = 16
explorer_batch = 64

log_name = "train_log/Train3_6/log_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file = open(log_name + ".txt", "wt")


history_learner = [[] for _ in range(task_num)]
history_policy =  [[] for _ in range(task_num)]

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

try:
    with sess.as_default():
        learner = Skill_Learner(state_len, action_len, goal_len, traj_len, traj_track_len, latent_len, latent_body_len, task_num, learner_unity=0.01, learner_diverse=0.001,
            learner_lr=0.0001, sampler_lr=0.0001, sampler_disperse=0.1)
        explorer = [Skill_Explorer(state_len,  action_len, goal_len, name=str(i)) for i in range(task_num)]
        learner_saver = tf.train.Saver(max_to_keep=0, var_list=learner.trainable_dict)
        explorer_savers = [tf.train.Saver(max_to_keep=0, var_list=explorer[i].trainable_dict) for i in range(task_num)]

        init = tf.global_variables_initializer()
        sess.run(init)

        learner.network_initialize()
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
        blueprints = [bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w")]
        for exp in range(1, 2001):
            cur_move = [0.] * task_num
            cur_reward = [0.] * task_num
            cur_move_num = [0.01] * task_num
            cur_reward_num = [0.01] * task_num
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
                if task == 0:
                    ratio_fl = 100.
                    ratio_fr = 100.
                    ratio_rl = 100.
                    ratio_rr = 100.
                if task == 1:
                    ratio_fl = 200.
                    ratio_fr = 50.
                    ratio_rl = 200.
                    ratio_rr = 50.
                if task == 2:
                    ratio_fl = 50.
                    ratio_fr = 200.
                    ratio_rl = 50.
                    ratio_rr = 200.

                states = []
                survive_vector = [True for _ in vehicles_list]
                broken_vector = [0 for _ in vehicles_list]
                broken_index = [0 for _ in vehicles_list]
                broken_value = [0.0 for _ in vehicles_list]
                first_traj_pos = []
                first_traj_yaw = []
                print( len(history_learner[task]))
                for step in range(1000):
                    
                    if step % traj_len == 0:
                        if (step // traj_len) % latent_preserve == 0:
                            desired_num = len(history_learner[task]) // 128
                            if desired_num > env_num:
                                desired_num = env_num
                            if desired_num > 0:
                                latent = np.random.normal(size=(desired_num, latent_body_len))
                                if desired_num != env_num:
                                    goal = np.concatenate([goal, 
                                        np.array([[[4., 0.], [8., 0.], [12., 0.], [16., 0.], [20., 0.]] for _ in range(env_num - desired_num)])])
                            else:
                                goal = np.array([[[4., 0.], [8., 0.], [12., 0.], [16., 0.], [20., 0.]] for _ in range(env_num)])
                        state_traj = [[] for _ in range(env_num)]
                        body_traj = [[] for _ in range(env_num)]
                        goal_traj = [[] for _ in range(env_num)]
                        action_traj = [[] for _ in range(env_num)]
                        current_goal = [goal[i][0] for i in range(env_num)]
                        prevscore =  [0. for _ in range(env_num)]

                        first_traj_pos = []
                        first_traj_yaw = []
                        relative_goal = []
                        states = []
                        for i, actor in enumerate(vehicles_list):
                            tr = actor.get_transform()
                            v = actor.get_velocity()
                            a = actor.get_acceleration()
                            yaw = tr.rotation.yaw * -0.017453293
                            yawsin = np.sin(yaw)
                            yawcos = np.cos(yaw)
                            vx, vy = rotate(v.x, v.y, yawsin, yawcos)
                            ax, ay = rotate(a.x, a.y, yawsin, yawcos)
                            states.append([0., 0., vx, vy, ax, ay, 1., 0., tr.rotation.roll, tr.rotation.pitch])
                            first_traj_pos.append([tr.location.x, tr.location.y])
                            first_traj_yaw.append([np.sin(tr.rotation.yaw * -0.017453293), np.cos(tr.rotation.yaw * -0.017453293)])  
                            d = np.sqrt(current_goal[i][0] ** 2 + current_goal[i][1] ** 2)
                            relative_goal.append([current_goal[i][0] / d, current_goal[i][1] / d])

                    goal_list = []
                    actions = explorer[task].get_action(states, relative_goal, False)

                    vehiclecontrols = []
                    for i, actor in enumerate(vehicles_list):

                        if broken_vector[i] == 0:
                            if random.random() < 0.005:
                                broken_vector[i] = 10
                                broken_index[i] = random.randrange(4)
                                broken_value[i] = random.random() * 0.5 - 0.2
                        else:  
                            broken_vector[i] -= 1
                            actions[i][broken_index[i]] = broken_value[i]

                            
                        control = carla.VehicleControl()
                        control.fl=float(actions[i][0] * ratio_fl)
                        control.fr=float(actions[i][1] * ratio_fr)
                        control.bl=float(actions[i][2] * ratio_rl)
                        control.br=float(actions[i][3] * ratio_rr)
                        control.gear = 1
                        control.manual_gear_shift = True
                        control.hand_brake = False
                        control.reverse = False
                        
                        vehiclecontrols.append(carla.command.ApplyVehicleControl(actor, control))

                    client.apply_batch(vehiclecontrols)
                    world.tick()

                    nextstates = []
                    next_relative_goal = []
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
                        nextstates.append([px, py, vx, vy, ax, ay, fx, fy, tr.rotation.roll, tr.rotation.pitch])
                        d = np.sqrt((current_goal[i][0] - px) ** 2 + (current_goal[i][1] - py) ** 2)
                        next_relative_goal.append([(current_goal[i][0] - px) / d, (current_goal[i][1] - py) / d])

                        #prevscore = np.sqrt((current_goal[i][0] - states[i][0]) ** 2 + (current_goal[i][1] - states[i][1]) ** 2)
                        #score = prevscore - np.sqrt((current_goal[i][0] - px) ** 2 + (current_goal[i][1] - py) ** 2)
                        score = 1. - ((relative_goal[i][0] * 0.25 - (px - states[i][0])) ** 2 + (relative_goal[i][1] * 0.25 - (py - states[i][1])) ** 2) * 4.



                        cur_reward[task] += score
                        cur_reward_num[task] += 1.
                        reward = ((actions[i][0] - actions[i][2]) ** 2 + (actions[i][1] - actions[i][3]) ** 2) * 0.1 
                        survive = (abs(tr.rotation.roll) < 20) and (abs(tr.rotation.pitch) < 20)
                        if survive_vector[i] :
                            if survive:
                                history_policy[task].append([states[i], nextstates[i][:], actions[i], relative_goal[i], [score - reward], [1.]])
                            else:
                                history_policy[task].append([states[i], nextstates[i][:], actions[i], relative_goal[i], [-10.], [0.]])
                                survive_vector[i] = survive

                        if step % traj_track_len == (traj_track_len - 1):
                            traj_tack_step = (step % traj_len) // traj_track_len
                            if traj_tack_step != (traj_len // traj_track_len - 1):
                                current_goal[i] = current_goal[i] - goal[i][traj_tack_step] +  goal[i][traj_tack_step + 1]


                        state_traj[i].append(states[i])
                        action_traj[i].append(actions[i])
                        if step % traj_track_len == traj_track_len - 1:
                            body_traj[i].append(np.array([px, py]))

                    states = nextstates
                    relative_goal = next_relative_goal
                    if step % traj_len == traj_len - 1:
                        for i, actor in enumerate(vehicles_list):
                            skill_moved = np.sqrt(body_traj[i][-1][0] ** 2 + body_traj[i][-1][1] ** 2)
                            if skill_moved > 1. and survive_vector[i]:
                                history_learner[task].append([state_traj[i], body_traj[i], action_traj[i]])
                            cur_move[task] += skill_moved
                            cur_move_num[task] += 1.
             
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
                cur_move[task] /= cur_move_num[task]
                cur_reward[task] /= cur_reward_num[task]
                print("Episode " + str(exp) + " Task " + str(task) +  " Move " +  str(cur_move[task] ) +  " Reward " +  str(cur_reward[task]) )

            train_learner = True
            for task in range(task_num):
                if len(history_learner[task]) < learner_batch * 8:
                    train_learner = False
            if train_learner:
                for iter in range(256):
                    dic = [random.sample(range(len(history_learner[task])), learner_batch) for task in range(task_num)]

                    state_dic = np.concatenate([[history_learner[task][x][0] for x in dic[task]] for task in range(task_num)])
                    body_dic = np.concatenate([[history_learner[task][x][1] for x in dic[task]] for task in range(task_num)])
                    action_dic = np.concatenate([[history_learner[task][x][2] for x in dic[task]] for task in range(task_num)])

                    learner.optimize(state_dic, body_dic, action_dic)
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
                    
            learner.log_print()
            for i in range(task_num):
                explorer[i].log_print()

            log_file.write(str(exp) + "\t")
            for i in range(task_num):
                log_file.write(str(cur_move[i])  + "\t" + str(cur_reward[i]) + "\t")
            log_file.write(learner.current_log())
            for i in range(task_num):
                log_file.write(explorer[i].current_log())
            log_file.write("\n")
            log_file.flush()

            learner.network_update()
            for i in range(task_num):
                explorer[i].network_update()

            if exp % 20 == 0:
                learner_saver.save(sess, log_name + "_learner_" + str(exp) + ".ckpt")
                for i in range(task_num):
                    explorer_savers[i].save(sess, log_name + "_explorer" + str(i) + "_" + str(exp) + ".ckpt")

            for task in range(task_num):
                history_learner[task] = history_learner[task][(len(history_learner[task]) // 32):]
                history_policy[task] = history_policy[task][(len(history_policy[task] ) // 32):]



finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)