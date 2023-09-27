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
mean_latent = [2.52, 0.39, -3.06]

def GetColor(z1, z2):
    z1 = z1 / 2.0
    z2 = z2 / 2.0
    color =  ( int( np.clip((0.5 + 1.28  * z2) * 255, 0, 255 ) ),
                int( np.clip((0.5 -  0.214 * z1 - 0.380 * z2) * 255, 0, 255 )),
                int( np.clip((0.5 + 2.128 * z1) * 255, 0, 255) ))
    return color

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos
  
traj_map = np.full((1024, 1024, 3), 255, np.uint8)
log_name = "test_log/Train2_1/log_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_traj_file = open(log_name + "traj.txt", "wt")

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
                    bp_library.find("vehicle.neubility.delivery_rightbig")]

        learner = Skill_Learner(state_len, action_len, goal_len, traj_len, traj_track_len, latent_len, latent_body_len, task_num)
        learner_saver = tf.train.Saver(max_to_keep=0, var_list=learner.trainable_dict)
        learner_saver.restore(sess, "train_log/Train2_1/log_2023-09-26-16-30-57_learner_250.ckpt")

        for task in range(task_num):
            log_traj_task_file = open(log_name + "traj_task_" + str(task) + ".txt", "wt")
            traj_task_map = np.full((1024, 1024, 3), 255, np.uint8)
            for zt_x_a, zt_y_a in itertools.product([-2., -1.2, -0.4, 0.4, 1.2], [-2., -1.2, -0.4, 0.4, 1.2]):
                vehicles_list = []
                latent = []
                action_latent = []
                for x, y in itertools.product(range(8), range(8)):                                                                                                                                                                                                                           
                    spawn_point = carla.Transform(carla.Location(x * 200. - 800., y * 200. - 800., 3.0), carla.Rotation(0, 0, 0))
                    actor = world.spawn_actor(blueprints[task], spawn_point)
                    vehicles_list.append(actor)

                    latent.append([zt_x_a + x * 0.1, zt_y_a + y * 0.1])
                    action_latent.append([zt_x_a + x * 0.1, zt_y_a + y * 0.1, mean_latent[task]])
                
                goal = learner.get_goal(latent, True) * 1.25
                if task == 0:
                    i = 0
                    for x, y in itertools.product(range(8), range(8)):
                        if x % 4 == 2 and y % 4 == 2:
                            cv2.polylines(traj_map, np.array([goal[i] * 20. + 512.], np.int32), False, GetColor(latent[i][0], latent[i][1]))
                            cv2.imwrite(log_name + "traj_map.png", traj_map)

                        log_traj_file.write(str(latent[i][0]) + "\t" + str(latent[i][1]))
                        for g in goal[i]:
                            log_traj_file.write("\t" + str(g[0]) + "\t" + str(g[1]))
                        log_traj_file.write("\n")
                        log_traj_file.flush()
                        i += 1
                

                for step in range(30):
                    world.tick()

                states = []
                first_traj_pos = []
                first_traj_yaw = []
                traj_pos = np.zeros((env_num, traj_len // traj_track_len, 2), np.float32)
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
                for step in range(traj_len * 5):
                    if step % traj_len == 0:
                        first_traj_pos = []
                        first_traj_yaw = []
                        for actor in vehicles_list:
                            tr = actor.get_transform()
                            first_traj_pos.append([tr.location.x, tr.location.y])
                            first_traj_yaw.append([np.sin(tr.rotation.yaw * -0.017453293), np.cos(tr.rotation.yaw * -0.017453293)])  

                    actions = learner.get_action(states, action_latent, True)

                    vehiclecontrols = []
                    for i, actor in enumerate(vehicles_list):
                        control = carla.VehicleControl()
                        wl = (actions[i][0] + actions[i][2]) / 2.
                        wr = (actions[i][1] + actions[i][3]) / 2.
                        control.fl=float(wl * (300.0 if task == 1 else 200.0))
                        control.fr=float(wr * (300.0 if task == 2 else 200.0))
                        control.bl=float(wl * (300.0 if task == 1 else 200.0))
                        control.br=float(wr * (300.0 if task == 2 else 200.0))
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
                        nextstates.append([float(step % traj_len + 1), px, py, vx, vy, ax, ay, fx, fy, tr.rotation.roll, tr.rotation.pitch])

                        if step % traj_track_len == (traj_track_len - 1):
                            traj_tack_step = (step % traj_len) // traj_track_len
                            traj_pos[i][traj_tack_step] += np.array([px, py * 1.5])

                    states = nextstates

                traj_pos /= 5
                i = 0
                for x, y in itertools.product(range(8), range(8)):
                    if x % 4 == 2 and y % 4 == 2:
                        cv2.polylines(traj_task_map, np.array([traj_pos[i] * 30. + 512.], np.int32), False, GetColor(latent[i][0], latent[i][1]))
                        cv2.imwrite(log_name + "traj_task_map_" + str(task) + ".png", traj_task_map)

                    log_traj_task_file.write(str(latent[i][0]) + "\t" + str(latent[i][1]))
                    for g in traj_pos[i]:
                        log_traj_task_file.write("\t" + str(g[0]) + "\t" + str(g[1]))
                    log_traj_task_file.write("\n")
                    log_traj_task_file.flush()
                    i += 1

                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
                print("Task " + str(task) +  " l1 " +  str(zt_x_a) +  " l2 " +  str(zt_y_a) )




finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)