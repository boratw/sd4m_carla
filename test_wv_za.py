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

import random
import pickle
from datetime import datetime
import itertools 
import cv2


tf.disable_eager_execution()
sess = tf.Session()

vehicles_list = []
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

agent_num = 150

state_len = 10
action_len = 4
goal_len = 2
traj_len = 100
traj_track_len = 20
latent_len = 3
latent_body_len = 2
latent_preserve = 4
task_num = 3
env_num = 64

learner_batch = 16

def GetColor(z1, z2):
    z1 = z1 / 2.0
    z2 = z2 / 2.0
    color =  ( int( np.clip((0.5 + 1.28  * z2) * 255, 0, 255 ) ),
                int( np.clip((0.5 -  0.214 * z1 - 0.380 * z2) * 255, 0, 255 )),
                int( np.clip((0.5 + 2.128 * z1) * 255, 0, 255) ))
    return color

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos
  
def get_state(i, actor):
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
    return [px, py, vx * 2., vy* 2., ax, ay, fx, fy, tr.rotation.roll, tr.rotation.pitch]

def get_control(actor, action, steer=5):
    control = carla.VehicleControl()
    if ratio_fl < 0.:
        action[0] = (action[1] + action[2] + action[3]) / 3
    if ratio_fr < 0.:
        action[1] = (action[0] + action[2] + action[3]) / 3
    if ratio_rl < 0.:
        action[2] = (action[0] + action[1] + action[3]) / 3
    if ratio_rr < 0.:
        action[3] = (action[1] + action[1] + action[2]) / 3
    action[0] = np.tanh((action[0] + 0.1) * (0.5 + steer * 0.025) + steer * 0.02) * 0.75
    action[1] = np.tanh((action[1] + 0.1) * (0.5 - steer * 0.025) - steer * 0.02) * 0.75
    action[2] = np.tanh((action[2] + 0.1) * (0.5 + steer * 0.025) + steer * 0.02) * 0.75
    action[3] = np.tanh((action[3] + 0.1) * (0.5 - steer * 0.025) - steer * 0.02) * 0.75
    #action = np.tanh(action)

    control.fl=float(action[0] * ratio_fl)
    control.fr=float(action[1] * ratio_fr)
    control.bl=float(action[2] * ratio_rl)
    control.br=float(action[3] * ratio_rr)
    control.gear = 1
    control.manual_gear_shift = True
    control.hand_brake = False
    control.reverse = False
    return carla.command.ApplyVehicleControl(actor, control)

log_name = "test_log/Train3_5/"
traj_map = np.full((1024, 1024, 3), 255, np.uint8)

latent_task = [[0.1, -3.4], #<- -1.0
    [-0.1, -3.6], [0.1, -3.3], [0.0, -3.4], [0.2, -3.1], [-0.1, -3.4], #<- -0.75
    [0.3, -2.9], [-0.2, -3.4], [-0.1, -3.2], [-0.2, -3.2], [-0.3, -3.2], #<- -0.5
    [-0.1, -2.9], [0.1, -2.6], [-0.1, -2.6], [-0.4, -2.6], [-0.1, -2.2], #<- -0.25
    [-0.5, -2.0], [-0.1, -1.6], [-0.9, -0.9], [-0.6, -0.3], [-0.5, 0.3], #<- 0.0
    [0.0, 0.2], [0.1, 0.6], [0.5, 0.4], [0.8, 0.1], [1.0, 0.9], #<- 0.25
    [1.2, 0.8], [1.4, 1.0], [1.4, 0.2], [1.7, 1.1], [1.7, 0.7], #<- 0.5
    [2.0, 1.4], [1.6, -0.2], [1.7, 0.0], [1.9, 0.5], [2.4, 1.7], #<- 0.75
    [2.1, 0.8], [2.1, 0.7], [1.8, -0.2], [2.0, 0.3], [1.7, -0.6]] #<- 1.0

try:
    with sess.as_default():
        world = client.get_world()
        settings = world.get_settings()
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        bp_library = world.get_blueprint_library()
        blueprints = [bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w"),
                    bp_library.find("vehicle.neubility.delivery4w")]

        learner = Skill_Learner(state_len, action_len, goal_len, traj_len, traj_track_len, latent_len, latent_body_len, task_num)
        learner_saver = tf.train.Saver(max_to_keep=0, var_list=learner.trainable_dict)
        learner_saver.restore(sess, "train_log/Train3_5/log_2023-10-31-23-53-09_learner_500.ckpt")

        for task in range(7, 11):
            ratio_fl = 100.
            ratio_fr = 100.
            ratio_rl = 100.
            ratio_rr = 100.
            yaw_diff = 0.
            steer = 0
            if task == 1:
                ratio_fl = 200.
                ratio_fr = 50.
                ratio_rl = 200.
                ratio_rr = 50.
            elif task == 2:
                ratio_fl = 50.
                ratio_fr = 200.
                ratio_rl = 50.
                ratio_rr = 200.
            elif task == 3:
                ratio_fl = -50.
            elif task == 4:
                ratio_fr = -50.
            elif task == 5:
                ratio_rl = -50.
            elif task == 6:
                ratio_rr = -50.
            elif task == 7:
                yaw_diff = -10.
                steer = 10
            elif task == 8:
                yaw_diff = 10.
                steer = -10
            elif task == 9:
                yaw_diff = -20.
                steer = 20
            elif task == 10:
                yaw_diff = 20.
                steer = -20



            #Finding Latent
            with open(log_name + "task_" + str(task) + "_score.txt", 'w') as f:
                za_list = [i * 0.1 - 10.0 for i in range(200)]
                for za in za_list:
                    env_num = 100
                    action_latent = []
                    vehicles_list = []
                    for x, y in itertools.product(range(10), range(10)):                                                                                                                                                                                                                           
                        spawn_point = carla.Transform(carla.Location(x * 200. - 1000., y * 200. - 1000., 3.0), carla.Rotation(0, 0, 0))
                        actor = world.spawn_actor(blueprints[task], spawn_point)
                        vehicles_list.append(actor)
                        action_latent.append([latent_task[steer + 20][0], latent_task[steer + 20][1], za + (x // 2) * 0.004 + (y // 2) * 0.02])
                    for step in range(30):
                        world.tick()
                    first_pos = []
                    first_traj_pos = []
                    first_traj_yaw = []
                    states = []
                    for i, actor in enumerate(vehicles_list):
                        tr = actor.get_transform()
                        first_pos.append([tr.location.x, tr.location.y])
                        first_traj_pos.append([tr.location.x, tr.location.y])
                        first_traj_yaw.append([np.sin((tr.rotation.yaw + yaw_diff) * -0.017453293), np.cos((tr.rotation.yaw + yaw_diff) * -0.017453293)])
                        states.append(get_state(i, actor))
                    wheel_output = [np.array([0., 0., 0., 0.]) for _ in vehicles_list]
                    for step in range(200):
                        actions = learner.get_action(states, action_latent, True)
                        vehiclecontrols = []
                        for i, actor in enumerate(vehicles_list):
                            wheel_output[i] += actions[i]
                            vehiclecontrols.append(get_control(actor, actions[i], steer))
                        client.apply_batch(vehiclecontrols)
                        world.tick()
                        
                        if step % traj_track_len == 0:
                            first_traj_pos = []
                            first_traj_yaw = []
                            for actor in vehicles_list:
                                tr = actor.get_transform()
                                first_traj_pos.append([tr.location.x, tr.location.y])
                                first_traj_yaw.append([np.sin((tr.rotation.yaw + yaw_diff) * -0.017453293), np.cos((tr.rotation.yaw + yaw_diff) * -0.017453293)])
                        states = []
                        for i, actor in enumerate(vehicles_list):
                            states.append(get_state(i, actor))
                        
                    for x, y in itertools.product(range(0, 10, 2), range(0, 10, 2)):
                        tx = 0
                        ty = 0
                        tx2 = 0
                        ty2 = 0
                        i = x + y * 10
                        w = np.array([0., 0., 0., 0.])
                        w2 = np.array([0., 0., 0., 0.])
                        for j in [i, i + 1, i + 10, i + 11]:
                            tr = vehicles_list[j].get_transform()
                            tx += (tr.location.x - first_pos[j][0])
                            ty += (tr.location.y - first_pos[j][1])
                            tx2 += (tr.location.x - first_pos[j][0]) ** 2
                            ty2 += (tr.location.y - first_pos[j][1]) ** 2
                            w += wheel_output[j]
                            w2 += wheel_output[j] ** 2
                        w /= 8
                        f.write(str(action_latent[i][2]) + "\t" + str(tx) + "\t" + str(tx2 - tx ** 2 / 4) + "\t" + str(ty) + "\t" + str(ty2 - ty ** 2 / 4) 
                            + "\t" + str(w[0]) + "\t" + str(w2[0] - w[0] ** 2 / 4) 
                            + "\t" + str(w[1]) + "\t" + str(w2[1] - w[1] ** 2 / 4) 
                            + "\t" + str(w[2]) + "\t" + str(w2[2] - w[2] ** 2 / 4) 
                            + "\t" + str(w[3]) + "\t" + str(w2[3] - w[3] ** 2 / 4) + "\n")
                    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
                        
            
                    print(task, za)



finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)