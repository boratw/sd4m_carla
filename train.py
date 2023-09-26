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

state_len = 6
action_len = 2
goal_len = 2
traj_len = 50
traj_track_len = 10
latent_len = 4
latent_body_len = 2
latent_preserve = 4
task_num = 15
env_num = 64

learner_batch = 16
explorer_batch = 64

log_name = "train_log/Train1/log_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file = open(log_name + ".txt", "wt")


history_learner = [[] for _ in range(task_num)]
history_policy =  [[] for _ in range(task_num)]

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

try:
    with sess.as_default():
        learner = Skill_Learner(state_len, action_len, goal_len, traj_len, traj_track_len, latent_len, latent_body_len, task_num,
            sampler_diverse=1., sampler_disperse=0., learner_lr=0.0001, sampler_lr=0.0001, learner_unity=0.1, learner_diverse=0.2)
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
        blueprints = [bp_library.find("vehicle.neubility.delivery"),
                    bp_library.find("vehicle.neubility.delivery_leftbig"),
                    bp_library.find("vehicle.neubility.delivery_rightbigbig")]
        for exp in range(1, 1000001):
            cur_move = [0.] * task_num
            cur_reward = [0.] * task_num
            for task in range(task_num):
                vehicles_list = []
                task_t = task % 5
                task_v = task // 5
                for x in range(8):
                    for y in range(8):
                        i = random.randrange(3)                                                                                                                                                                                                                                                                      
                        spawn_point = carla.Transform(carla.Location(x * 200. - 800., y * 200. - 800., 3.0), carla.Rotation(0, 0, 0))
                        actor = world.spawn_actor(blueprints[task_v], spawn_point)
                        vehicles_list.append(actor)

                        
                for step in range(30):
                    world.tick()

                states = []
                first_traj_pos = []
                first_traj_yaw = []
                for actor in vehicles_list:
                    v = actor.get_velocity()
                    a = actor.get_acceleration()
                    yaw = actor.get_transform().rotation.yaw * -0.017453293
                    yawsin = np.sin(yaw)
                    yawcos = np.cos(yaw)
                    vx, vy = rotate(v.x, v.y, yawsin, yawcos)
                    ax, ay = rotate(a.x, a.y, yawsin, yawcos)
                    states.append([vx, vy, ax, ay, 0., 0.])
                print( len(history_learner[task]))
                for step in range(500):
                    
                    if step % traj_len == 0:
                        if (step // traj_len) % latent_preserve == 0:
                            desired_num = len(history_learner[task]) // 256
                            if desired_num > env_num:
                                desired_num = env_num
                            if desired_num > 0:
                                latent = np.random.normal(size=(desired_num, latent_body_len))
                                goal = learner.get_goal_batch(latent, True)
                                if desired_num != env_num:
                                    goal = np.concatenate([goal, 
                                        np.array([[[4., 0.], [8., 0.], [12., 0.], [16., 0.], [20., 0.]] for _ in range(env_num - desired_num)])])
                            else:
                                goal = np.array([[[4., 0.], [8., 0.], [12., 0.], [16., 0.], [20., 0.]] for _ in range(env_num)])
                            goal_swapped = np.array(np.swapaxes(goal,0,1))
                        state_traj = [[] for _ in range(env_num)]
                        body_traj = [[] for _ in range(env_num)]
                        action_traj = [[] for _ in range(env_num)]
                        current_goal = np.zeros((env_num, 2), np.float32)

                        first_traj_pos = []
                        first_traj_yaw = []
                        for actor in vehicles_list:
                            tr = actor.get_transform()
                            first_traj_pos.append([tr.location.x, tr.location.y])
                            first_traj_yaw.append([np.sin(tr.rotation.yaw * -0.017453293), np.cos(tr.rotation.yaw * -0.017453293)])  
                    if step % traj_track_len == 0:
                        current_goal = goal_swapped[(step % traj_len) // traj_track_len]
                        current_goal_rel = current_goal.copy()

                    actions = explorer[task].get_action_batch(states, current_goal_rel, False)

                    vehiclecontrols = []
                    for i, actor in enumerate(vehicles_list):

                        dropout = random.randrange(20)
                        if dropout < action_len:
                            actions[i][dropout] = np.tanh(actions[i][dropout] + np.random.normal(0., 0.5))

                        v = actor.get_velocity()
                        vs = np.sqrt(v.x * v.x + v.y * v.y)
                        control = carla.VehicleControl()
                        control.steer = float(actions[i][0])
                        if actions[i][1] > 0.:
                            control.throttle = float(actions[i][1])
                            control.brake = 0.
                        else:
                            control.throttle = 0.
                            control.brake = -(float(actions[i][1]) * 0.7)
                        control.manual_gear_shift = False
                        control.hand_brake = False
                        control.reverse = False
                        control.gear = 0
                        
                        if task_t == 1:
                            control.steer /= 1.414213
                            torque = carla.Vector3D(0.0, 0.0, 2e8 * vs)
                            vehiclecontrols.append(carla.command.ApplyTorque(actor, torque))
                            vehiclecontrols.append(carla.command.ApplyImpulse(actor, v * -4))
                        elif task_t == 2:
                            control.steer /= 1.414213
                            torque = carla.Vector3D(0.0, 0.0, -2e8 * vs)
                            vehiclecontrols.append(carla.command.ApplyTorque(actor, torque))
                            vehiclecontrols.append(carla.command.ApplyImpulse(actor, v * -4))
                        elif task_t == 3:
                            control.throttle /= 1.414213
                            torque = carla.Vector3D(0.0, 0.0, 2e8 * vs + 1e9 * control.throttle)
                            vehiclecontrols.append(carla.command.ApplyTorque(actor, torque))
                            vehiclecontrols.append(carla.command.ApplyImpulse(actor, v * -4))
                        elif task_t == 4:
                            control.throttle /= 1.414213
                            torque = carla.Vector3D(0.0, 0.0, -2e8 * vs + -1e9 * control.throttle)
                            vehiclecontrols.append(carla.command.ApplyTorque(actor, torque))
                            vehiclecontrols.append(carla.command.ApplyImpulse(actor, v * -4))
                        vehiclecontrols.append(carla.command.ApplyVehicleControl(actor, control))

                    client.apply_batch(vehiclecontrols)
                    world.tick()

                    nextstates = []
                    for i, actor in enumerate(vehicles_list):
                        tr = actor.get_transform()
                        v = actor.get_velocity()
                        a = actor.get_acceleration()
                        yaw = actor.get_transform().rotation.yaw * -0.017453293
                        yawsin = np.sin(yaw)
                        yawcos = np.cos(yaw)
                        vx, vy = rotate(v.x, v.y, yawsin, yawcos)
                        ax, ay = rotate(a.x, a.y, yawsin, yawcos)
                        nextstates.append([vx, vy, ax, ay, actions[i][0], actions[i][1]])

                        dx, dy = rotate(tr.location.x - first_traj_pos[i][0], tr.location.y - first_traj_pos[i][1], first_traj_yaw[i][0], first_traj_yaw[i][1])
                        prevscore = np.sqrt(current_goal_rel[i][0] * current_goal_rel[i][0] + current_goal_rel[i][1] * current_goal_rel[i][1])
                        current_goal_rel[i] = current_goal[i] - np.array([dx, dy])
                        score = np.sqrt(current_goal_rel[i][0] * current_goal_rel[i][0] + current_goal_rel[i][1] * current_goal_rel[i][1])
                        history_policy[task].append([states[i], nextstates[i], actions[i], current_goal[i], [(prevscore - score) * 20.]])
                        cur_reward[task] += (prevscore - score)

                        state_traj[i].append(states[i])
                        action_traj[i].append(actions[i])
                        if step % traj_track_len == traj_track_len - 1:
                            if len(body_traj[i]) == 0:
                                body_traj[i].append(np.array([dx, dy]))
                            else:
                                body_traj[i].append(np.array([dx, dy]) + body_traj[i][-1])
                    states = nextstates
                    if step % traj_len == traj_len - 1:
                        for i, actor in enumerate(vehicles_list):
                            skill_moved = np.sqrt(body_traj[i][-1][0] ** 2 + body_traj[i][-1][1] ** 2)
                            if skill_moved > 1.:
                                history_learner[task].append([state_traj[i], body_traj[i], action_traj[i]])
                            cur_move[task] += skill_moved
             
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
                cur_move[task] /= env_num
                cur_reward[task] /= env_num
                print("Episode " + str(exp) + " Task " + str(task) +  " Move " +  str(cur_move[task] ) +  " Reward " +  str(cur_reward[task]) )

            if np.array([len(history_learner[task]) for task in range(task_num)]).all() > learner_batch:
                for iter in range(256):
                    dic = [random.sample(range(len(history_learner[task])), learner_batch) for task in range(task_num)]

                    state_dic = np.concatenate([[history_learner[task][x][0] for x in dic[task]] for task in range(task_num)])
                    body_dic = np.concatenate([[history_learner[task][x][1] for x in dic[task]] for task in range(task_num)])
                    action_dic = np.concatenate([[history_learner[task][x][2] for x in dic[task]] for task in range(task_num)])

                    learner.optimize_batch(state_dic, body_dic, action_dic)
            for iter in range(32):
                for iter2 in range(32):
                    for task in range(task_num):
                        if len(history_policy[task]) > explorer_batch:
                            dic = random.sample(range(len(history_policy[task])), explorer_batch)

                            state_dic = [history_policy[task][x][0] for x in dic]
                            nextstate_dic = [history_policy[task][x][1] for x in dic]
                            action_dic = [history_policy[task][x][2] for x in dic]
                            goal_dic = [history_policy[task][x][3] for x in dic]
                            reward_dic = [history_policy[task][x][4] for x in dic]

                            explorer[task].optimize_batch(state_dic, nextstate_dic, action_dic, goal_dic, reward_dic, exp)
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