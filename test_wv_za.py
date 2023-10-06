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
    return [float(step % traj_len), px, py, vx, vy, ax, ay, fx, fy, tr.rotation.roll, tr.rotation.pitch]

def get_control(actor, action):
    control = carla.VehicleControl()
    wl = (action[0] + action[2]) / 2.
    wr = (action[1] + action[3]) / 2.
    

    control.fl=float(wl * ratio_fl)
    control.fr=float(wr * ratio_fr)
    control.bl=float(wl * ratio_rl)
    control.br=float(wr * ratio_rr)
    control.gear = 1
    control.manual_gear_shift = True
    control.hand_brake = False
    control.reverse = False
    return carla.command.ApplyVehicleControl(actor, control)

log_name = "test_log/Train2_1/"
log_latent_file = open(log_name + "latent.txt", "wt")
log_traj_file = open(log_name + "traj.txt", "wt")
traj_map = np.full((1024, 1024, 3), 255, np.uint8)

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
        blueprints = [bp_library.find("vehicle.neubility.delivery"),
                    bp_library.find("vehicle.neubility.delivery_leftbig"),
                    bp_library.find("vehicle.neubility.delivery_rightbig"),
                    bp_library.find("vehicle.neubility.delivery"),
                    bp_library.find("vehicle.neubility.delivery"),
                    bp_library.find("vehicle.neubility.delivery"),
                    bp_library.find("vehicle.neubility.delivery"),
                    bp_library.find("vehicle.neubility.delivery_nofl"),
                    bp_library.find("vehicle.neubility.delivery_nofr"),
                    bp_library.find("vehicle.neubility.delivery_norl"),
                    bp_library.find("vehicle.neubility.delivery_norr")]

        learner = Skill_Learner(state_len, action_len, goal_len, traj_len, traj_track_len, latent_len, latent_body_len, task_num)
        learner_saver = tf.train.Saver(max_to_keep=0, var_list=learner.trainable_dict)
        learner_saver.restore(sess, "train_log/Train2_1/log_2023-09-26-16-30-57_learner_250.ckpt")

        for task in range(11):
            ratio_fl = 200.
            ratio_fr = 200.
            ratio_rl = 200.
            ratio_rr = 200.
            if task == 1:
                ratio_fl = 300.
                ratio_rl = 300.
            elif task == 2:
                ratio_fr = 300.
                ratio_rr = 300.
            elif task == 3 or task == 7:
                ratio_fl = 0.
            elif task == 4 or task == 8:
                ratio_fr = 0.
            elif task == 5 or task == 9:
                ratio_rl = 0.
            elif task == 6 or task == 10:
                ratio_rr = 0.


            #Finding Latent
            maxlatent = 0
            maxscore = -9999
            with open(log_name + "task_" + str(task) + "_score.txt", 'w') as f:
                for za in [-10.0, -7.5, -5., -2.5, 0., 2.5, 5., 7.5]:
                    env_num = 100
                    action_latent = []
                    vehicles_list = []
                    for x, y in itertools.product(range(10), range(10)):                                                                                                                                                                                                                           
                        spawn_point = carla.Transform(carla.Location(x * 200. - 1000., y * 200. - 1000., 3.0), carla.Rotation(0, 0, 0))
                        actor = world.spawn_actor(blueprints[task], spawn_point)
                        vehicles_list.append(actor)
                        action_latent.append([0.3, -0.3, za + (x // 2) * 0.1 + (y // 2) * 0.5])
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
                        first_traj_yaw.append([np.sin(tr.rotation.yaw * -0.017453293), np.cos(tr.rotation.yaw * -0.017453293)])
                        states.append(get_state(i, actor))
                    for step in range(500):
                        actions = learner.get_action(states, action_latent, True)
                        vehiclecontrols = []
                        for i, actor in enumerate(vehicles_list):
                            vehiclecontrols.append(get_control(actor, actions[i]))
                        client.apply_batch(vehiclecontrols)
                        world.tick()
                        
                        if step % traj_len == 0:
                            first_traj_pos = []
                            first_traj_yaw = []
                            for actor in vehicles_list:
                                tr = actor.get_transform()
                                first_traj_pos.append([tr.location.x, tr.location.y])
                                first_traj_yaw.append([np.sin(tr.rotation.yaw * -0.017453293), np.cos(tr.rotation.yaw * -0.017453293)])
                        states = []
                        for i, actor in enumerate(vehicles_list):
                            states.append(get_state(i, actor))
                        
                    for x, y in itertools.product(range(0, 10, 2), range(0, 10, 2)):
                        tx = 0
                        ty = 0
                        i = x + y * 10
                        for j in [i, i + 1, i + 10, i + 11]:
                            tr = vehicles_list[j].get_transform()
                            tx += (tr.location.x - first_pos[j][0])
                            ty += (tr.location.y - first_pos[j][1])
                        score = tx - abs(ty) * 2
                        if score > maxscore:
                            maxscore = score
                            maxlatent = action_latent[i][2]
                        f.write(str(action_latent[i][2]) + "\t" + str(score) + "\t" + str(tx) + "\t" + str(ty) + "\n")
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