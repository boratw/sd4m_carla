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
    return [float(step % traj_track_len), px, py, vx, vy, ax, ay, fx, fy, tr.rotation.roll, tr.rotation.pitch]

def get_control(actor, action):
    control = carla.VehicleControl()
    control.fl=float(action[0] * ratio_fl)
    control.fr=float(action[1] * ratio_fr)
    control.bl=float(action[2] * ratio_rl)
    control.br=float(action[3] * ratio_rr)
    control.gear = 1
    control.manual_gear_shift = True
    control.hand_brake = False
    control.reverse = False
    return carla.command.ApplyVehicleControl(actor, control)

log_name = "test_log/Train3_6_2/"
log_latent_file = open(log_name + "latent.txt", "wt")
log_traj_file = open(log_name + "traj.txt", "wt")
traj_map = np.full((1024, 1024, 3), 255, np.uint8)

maxlatent = [0.24, 0.04, -2.04, -1.28, 0.32, -3.52, 0.4, -1.84, 0.44, -7.2, 8.5]

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
                    bp_library.find("vehicle.neubility.delivery4w")]

        learner = Skill_Learner(state_len, action_len, goal_len, traj_len, traj_track_len, latent_len, latent_body_len, task_num)
        learner_saver = tf.train.Saver(max_to_keep=0, var_list=learner.trainable_dict)
        learner_saver.restore(sess, "train_log/Train3_6_2/log_2023-11-14-18-00-49_learner_440.ckpt")
        for task in range(9):
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
            elif task == 3:
                ratio_fl = -50.
            elif task == 4:
                ratio_fr = -50.
            elif task == 5:
                ratio_rl = -50.
            elif task == 6:
                ratio_rr = -50.
            elif task == 7:
                ratio_fl = 0.
            elif task == 8:
                ratio_fr = 0.
            elif task == 9:
                ratio_fl = 0.
            elif task == 10:
                ratio_fr = 0.

            vehicles_list = []

            env_num = 100
            for za in [0.24, -2.4, 1.8, maxlatent[task]]:
                traj_task_map = np.full((1024, 1024, 3), 255, np.uint8)
                log_traj_task_file = open(log_name + "traj_task_%d_%.2f.txt" % (task, za), "wt")
                for zt_x_a, zt_y_a in itertools.product([-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5],
                    [-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5]):
                    vehicles_list = []
                    latent = []
                    action_latent = []
                    for x, y in itertools.product(range(10), range(10)):  
                                                                                                                                                                                                               
                        #spawn_point = carla.Transform(carla.Location(x * 200. - 800., y * 200. - 800., 3.0), carla.Rotation(0, 0, 0))
                        #actor = world.spawn_actor(blueprints[task], spawn_point)
                        #vehicles_list.append(actor)


                        latent.append([zt_x_a + (x // 2) * 0.1, zt_y_a + (y // 2) * 0.1])
                        action_latent.append([zt_x_a + (x // 2) * 0.1, zt_y_a + (y // 2) * 0.1, za])
                    if task == 0 and za == 0.24:
                        goal = learner.get_goal(latent, True)
                        goal_traj = np.zeros((env_num, traj_len // traj_track_len + 1, 2), np.float32)
                        goal_traj[:, 1:] = goal
                        #goal_traj *= np.array([1., 1.5])
                        for x, y in itertools.product(range(0, 10, 2), range(0, 10, 2)):
                            i = x + y * 10
                            #traj_map_ind = np.full((1024, 1024, 3), 0, np.uint8)
                            if i % 8 == 0:
                                r = np.array([(goal_traj[i] + goal_traj[i + 1] + goal_traj[i + 10] + goal_traj[i + 11]) * 5. + 512.], np.int32)
                                cv2.polylines(traj_map, r, False, GetColor(latent[i][0], latent[i][1]))
                            #cv2.polylines(traj_map_ind, r, False, (255, 255, 255))
                            #cv2.imwrite(log_name + "traj_map_%.2f_%.2f.png" % (latent[i][0], latent[i][1]), traj_map_ind)

                            log_traj_file.write(str(latent[i][0]) + "\t" + str(latent[i][1]))
                            for g in goal[i]:
                                log_traj_file.write("\t" + str(g[0]) + "\t" + str(g[1]))
                            log_traj_file.write("\n")
                            log_traj_file.flush()
                cv2.imwrite(log_name + "traj_map.png", traj_map)
                    
                '''
                    for step in range(30):
                        world.tick()

                    states = []
                    first_traj_pos = []
                    first_traj_yaw = []
                    traj_pos = np.zeros((env_num, traj_len // traj_track_len + 1, 2), np.float32)
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
                            vehiclecontrols.append(get_control(actor, actions[i]))

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
                                traj_tack_step = (step % traj_len) // traj_track_len + 1
                                traj_pos[i][traj_tack_step] += np.array([px, py])

                        states = nextstates

                    traj_pos /= 5
                    i = 0
                    for x, y in itertools.product(range(0, 10, 2), range(0, 10, 2)):
                        i = x + y * 10
                        traj_task_map_ind = np.full((1024, 1024, 3), 0, np.uint8)
                        r = np.array([(traj_pos[i] + traj_pos[i + 1] + traj_pos[i + 10] + traj_pos[i + 11]) * 10. + 512.], np.int32)
                        cv2.polylines(traj_task_map, r, False, GetColor(latent[i][0], latent[i][1]))
                        cv2.polylines(traj_task_map_ind, r, False, (255, 255, 255))
                        cv2.imwrite(log_name + "traj_task_map_%d_%.2f_%.2f_%.2f.png" % (task, za, latent[i][0], latent[i][1]), traj_task_map_ind)


                        log_traj_task_file.write(str(latent[i][0]) + "\t" + str(latent[i][1]))
                        for g in traj_pos[i]:
                            log_traj_task_file.write("\t" + str(g[0]) + "\t" + str(g[1]))
                        log_traj_task_file.write("\n")
                        log_traj_task_file.flush()
                        i += 1

                    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
                    print("Task " + str(task) +  "_" + str(za) + " l1 " +  str(zt_x_a) +  " l2 " +  str(zt_y_a) )
                    '''
                #cv2.imwrite(log_name + "traj_task_map_%d_%.2f.png" % (task, za), traj_task_map)
                



finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)