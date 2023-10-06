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
import itertools 
import cv2
import json


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
task_num = 11
env_num = 100


route_straight = [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0], [12.0, 0.0], [14.0, 0.0], [16.0, 0.0], [18.0, 0.0], [20.0, 0.0], [22.0, 0.0], [24.0, 0.0], [26.0, 0.0], [28.0, 0.0], [30.0, 0.0], [32.0, 0.0], [34.0, 0.0], [36.0, 0.0], [38.0, 0.0], [40.0, 0.0], [42.0, 0.0], [44.0, 0.0], [46.0, 0.0], [48.0, 0.0], [50.0, 0.0], [52.0, 0.0], [54.0, 0.0], [56.0, 0.0], [58.0, 0.0], [60.0, 0.0], [62.0, 0.0], [64.0, 0.0], [66.0, 0.0], [68.0, 0.0], [70.0, 0.0], [72.0, 0.0], [74.0, 0.0], [76.0, 0.0], [78.0, 0.0], [80.0, 0.0], [82.0, 0.0], [84.0, 0.0], [86.0, 0.0], [88.0, 0.0], [90.0, 0.0], [92.0, 0.0], [94.0, 0.0], [96.0, 0.0], [98.0, 0.0], [100.0, 0.0], [102.0, 0.0], [104.0, 0.0], [106.0, 0.0], [108.0, 0.0], [110.0, 0.0], [112.0, 0.0], [114.0, 0.0], [116.0, 0.0], [118.0, 0.0], [120.0, 0.0], [122.0, 0.0], [124.0, 0.0], [126.0, 0.0], [128.0, 0.0], [130.0, 0.0], [132.0, 0.0], [134.0, 0.0], [136.0, 0.0], [138.0, 0.0], [140.0, 0.0], [142.0, 0.0], [144.0, 0.0], [146.0, 0.0], [148.0, 0.0], [150.0, 0.0], [152.0, 0.0], [154.0, 0.0], [156.0, 0.0], [158.0, 0.0], [160.0, 0.0]]
route_leftturn = [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0], [12.0, 0.0], [14.0, 0.0], [16.0, 0.0], [18.0, 0.0], [20.0, 0.0], [22.0, 0.0], [24.0, 0.0], [26.0, 0.0], [28.0, 0.0], [30.0, 0.0], [32.0, 0.0], [34.0, 0.0], [36.0, 0.0], [38.0, 0.0], [40.0, 0.0], [42.0, 0.0], [44.0, 0.0], [46.0, 0.0], [48.0, 0.0], [50.0, 0.0], [52.0, 0.0], [54.0, 0.0], [56.0, 0.0], [58.0, 0.0], [60.0, 0.0], [62.0, 0.0], [64.0, 0.0], [66.0, 0.0], [68.0, 0.0], [70.0, 0.0], [72.0, 0.0], [74.0, 0.0], [76.0, 0.0], [78.0, 0.0], [80.0, 0.0], [82.0, 0.0], [84.0, 0.0], [86.0, 0.0], [88.0, 0.0], [90.0, 0.0], [92.0, 0.0], [94.0, 0.0], [95.9972590693584, -0.10467191536228049], [97.98235137182199, -0.3484106088432736], [99.93864657089418, -0.7642340017483837], [101.84075959814405, -1.3822680069344768], [103.6533751620733, -2.2275045521698322], [105.31145028892918, -3.345890386174309], [106.72566382075165, -4.760103979098025], [107.84404958311882, -6.4181791542738695], [108.68928605003993, -8.230794754721664], [109.30731997304503, -10.132907808673748], [109.7231432814282, -12.089203025711624], [109.96688188914307, -14.074295338705964], [110.07155371821356, -16.071554412586725], [110.07155363180335, -18.07155441258672], [110.07155354539314, -20.07155441258672], [110.07155345898293, -22.07155441258672], [110.07155337257272, -24.07155441258672], [110.07155328616251, -26.07155441258672], [110.0715531997523, -28.07155441258672], [110.07155311334209, -30.07155441258672], [110.07155302693188, -32.07155441258672], [110.07155294052167, -34.07155441258672], [110.07155285411146, -36.07155441258672], [110.07155276770125, -38.07155441258672], [110.07155268129104, -40.07155441258672], [110.07155259488083, -42.07155441258672], [110.07155250847062, -44.07155441258672], [110.0715524220604, -46.07155441258672], [110.0715523356502, -48.07155441258672], [110.07155224923999, -50.07155441258672], [110.07155216282978, -52.07155441258672], [110.07155207641956, -54.07155441258672], [110.07155199000935, -56.07155441258672]]
route_rightturn = [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0], [12.0, 0.0], [14.0, 0.0], [16.0, 0.0], [18.0, 0.0], [20.0, 0.0], [22.0, 0.0], [24.0, 0.0], [26.0, 0.0], [28.0, 0.0], [30.0, 0.0], [32.0, 0.0], [34.0, 0.0], [36.0, 0.0], [38.0, 0.0], [40.0, 0.0], [42.0, 0.0], [44.0, 0.0], [46.0, 0.0], [48.0, 0.0], [50.0, 0.0], [52.0, 0.0], [54.0, 0.0], [56.0, 0.0], [58.0, 0.0], [60.0, 0.0], [62.0, 0.0], [64.0, 0.0], [66.0, 0.0], [68.0, 0.0], [70.0, 0.0], [72.0, 0.0], [74.0, 0.0], [76.0, 0.0], [78.0, 0.0], [80.0, 0.0], [82.0, 0.0], [84.0, 0.0], [86.0, 0.0], [88.0, 0.0], [90.0, 0.0], [92.0, 0.0], [94.0, 0.0], [95.99238939576509, 0.1743114902776158], [97.93298084506526, 0.6581552945192664], [99.74559640899452, 1.5033918397546215], [101.32161789374598, 2.7347148191559665], [102.55294080505516, 4.310736357106862], [103.39817727197627, 6.1233519575546556], [103.88202099237446, 8.063943427759353], [104.0563323965707, 10.056332831055592], [104.05633231016049, 12.05633283105559], [104.05633222375027, 14.056332831055588], [104.05633213734006, 16.056332831055585], [104.05633205092985, 18.056332831055585], [104.05633196451964, 20.056332831055585], [104.05633187810943, 22.056332831055585], [104.05633179169922, 24.056332831055585], [104.05633170528901, 26.056332831055585], [104.0563316188788, 28.056332831055585], [104.05633153246859, 30.056332831055585], [104.05633144605838, 32.056332831055585], [104.05633135964817, 34.056332831055585], [104.05633127323796, 36.056332831055585], [104.05633118682775, 38.056332831055585], [104.05633110041754, 40.056332831055585], [104.05633101400733, 42.056332831055585], [104.05633092759712, 44.056332831055585], [104.0563308411869, 46.056332831055585], [104.0563307547767, 48.056332831055585], [104.05633066836648, 50.056332831055585], [104.05633058195627, 52.056332831055585], [104.05633049554606, 54.056332831055585], [104.05633040913585, 56.056332831055585], [104.05633032272564, 58.056332831055585], [104.05633023631543, 60.056332831055585]]
route_uturn = [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0], [12.0, 0.0], [14.0, 0.0], [16.0, 0.0], [18.0, 0.0], [20.0, 0.0], [22.0, 0.0], [24.0, 0.0], [26.0, 0.0], [28.0, 0.0], [30.0, 0.0], [32.0, 0.0], [34.0, 0.0], [36.0, 0.0], [38.0, 0.0], [40.0, 0.0], [42.0, 0.0], [44.0, 0.0], [46.0, 0.0], [48.0, 0.0], [50.0, 0.0], [52.0, 0.0], [54.0, 0.0], [56.0, 0.0], [58.0, 0.0], [60.0, 0.0], [62.0, 0.0], [64.0, 0.0], [66.0, 0.0], [68.0, 0.0], [70.0, 0.0], [72.0, 0.0], [74.0, 0.0], [76.0, 0.0], [78.0, 0.0], [80.0, 0.0], [82.0, 0.0], [84.0, 0.0], [86.0, 0.0], [88.0, 0.0], [90.0, 0.0], [92.0, 0.0], [94.0, 0.0], [95.98904379013439, -0.2090569322644298], [97.86842902513864, -0.893097236959997], [99.40051788669064, -2.1786724857526574], [100.24575435361174, -3.991288086200451], [100.24575426720153, -5.991288086200449], [99.40051764365194, -7.803903613611179], [97.86842867101312, -9.089478730015719], [95.98904337690081, -9.773518872313215], [93.99999956870177, -9.982575632703957], [91.99999956870178, -9.982575459883543], [89.9999995687018, -9.98257528706313], [87.99999956870181, -9.982575114242715], [85.99999956870182, -9.982574941422302], [83.99999956870184, -9.982574768601888], [81.99999956870185, -9.982574595781474], [79.99999956870187, -9.98257442296106], [77.99999956870188, -9.982574250140646], [75.9999995687019, -9.982574077320232], [73.99999956870191, -9.982573904499818], [71.99999956870192, -9.982573731679404], [69.99999956870194, -9.98257355885899], [67.99999956870195, -9.982573386038576], [65.99999956870197, -9.982573213218162], [63.99999956870197, -9.982573040397748], [61.99999956870198, -9.982572867577334], [59.99999956870199, -9.98257269475692], [57.999999568701995, -9.982572521936506], [55.999999568702, -9.982572349116092], [53.99999956870201, -9.982572176295678], [51.999999568702016, -9.982572003475264], [49.99999956870202, -9.98257183065485], [47.99999956870203, -9.982571657834436], [45.99999956870204, -9.982571485014022]]


#-1.0, -1.4, ...., 1.0
latent_task = [[-2.0, -2.0], #<- -1.0
    [-1.8, -1.9], [-1.8, -1.4], [-1.5, -1.6], [-1.2, -1.6], [-0.8, -1.7], #<- -0.5
    [-0.3, -1.8], [0.3, -1.8], [0.8, -1.6], [1.6, -1.4], [1.5, -0.8], #<- 0.0
    [1.2, -0.2], [1.1, 0.2], [1.5, 0.5], [0.9, 0.9], [1.3, 1.2], #<-0.5
    [1.3, 1.5], [1.3, 1.8], [0.6, 1.9], [0.0, 1.9], [-0.5, 1.9]] #<- 1.0

maxlatent = [-1.1, -2.3, 1.8, -4.0, 0.2, -7.0, 0.3, -5.7, -1.6, -7.2, 8.5]

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos
  
log_name = "test_log/Train2_1/"

def get_state(i, actor):
    tr = actor.get_transform()
    v = actor.get_velocity()
    a = actor.get_acceleration()
    tr = actor.get_transform()
    yaw = tr.rotation.yaw * -0.017453293
    yawsin = np.sin(yaw)
    yawcos = np.cos(yaw)
    f = tr.get_forward_vector()
    vx, vy = rotate(v.x, v.y, yawsin, yawcos)
    ax, ay = rotate(a.x, a.y, yawsin, yawcos)
    fx, fy = rotate(f.x, f.y, yawsin, yawcos)
    if controller < 3:
        return [25., 0., 0., vx, vy, ax, ay, fx, fy, tr.rotation.roll, tr.rotation.pitch]
    else:
        return [5., 0., 0., vx, vy, ax, ay, fx, fy, tr.rotation.roll, tr.rotation.pitch]


def get_control(actor, action, control):
    if controller < 3:
        action = np.tanh(action)
        control.fl=float(action[0] * ratio_fl)
        control.fr=float(action[1] * ratio_fr)
        control.bl=float(action[2] * ratio_rl)
        control.br=float(action[3] * ratio_rr)
    else:
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
        explorer = [Skill_Explorer(state_len,  action_len, goal_len, name=str(i)) for i in range(task_num)]
        explorer_savers = [tf.train.Saver(max_to_keep=0, var_list=explorer[i].trainable_dict) for i in range(task_num)]
        explorer_savers[0].restore(sess, "train_log/Train2_1/log_2023-09-26-16-30-57_explorer0_250.ckpt")
        explorer_savers[1].restore(sess, "train_log/Train2_1/log_2023-09-26-16-30-57_explorer1_250.ckpt")
        explorer_savers[2].restore(sess, "train_log/Train2_1/log_2023-09-26-16-30-57_explorer2_250.ckpt")

        latent_for_task = []
        for task in range(task_num):
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

            env_num = 25
            for route_it, route in enumerate([route_straight, route_leftturn, route_rightturn, route_uturn]): 
                for controller in range(5):
                    log_pos = [[] for _ in range(env_num)]
                    log_vel = [[] for _ in range(env_num)]
                    log_distance = [[] for _ in range(env_num)]
                    log_routeit = [[] for _ in range(env_num)]
                    log_wheel = [[] for _ in range(env_num)]
                    log_action = [[] for _ in range(env_num)]

                    vehicles_list = []
                    for x, y in itertools.product(range(5), range(5)):                                                                                                                                                                                                                           
                        spawn_point = carla.Transform(carla.Location(x * 400. - 1000., y * 400. - 1000., 3.0), carla.Rotation(0, 0, 0))
                        actor = world.spawn_actor(blueprints[task], spawn_point)
                        vehicles_list.append(actor)
                    for step in range(30):
                        world.tick()

                    first_pos = []
                    states = []
                    action_latent = []
                    goals = []
                    for i, actor in enumerate(vehicles_list):
                        tr = actor.get_transform()
                        first_pos.append([tr.location.x, tr.location.y])
                        states.append(get_state(i, actor))
                        if controller == 3:
                            action_latent.append([1.5, -0.8, -1.1])
                        else:
                            action_latent.append([1.5, -0.8, maxlatent[task]])
                        goals.append([2.0, 0.0])

                    route_target_iter = [0 for _ in range(env_num)]
                    route_current_iter = [0 for _ in range(env_num)]
                    for step in range(1500):
                        if controller < 3:
                            actions = explorer[controller].get_action(states, goals, True)
                        else:
                            print(action_latent[0])
                            actions = learner.get_action(states, action_latent, True)
                        vehiclecontrols = []
                        for i, actor in enumerate(vehicles_list):
                            control = actor.get_control()
                            log_wheel[i].append([control.fl, control.fr, control.bl, control.br])
                            log_action[i].append([float(actions[i][0]), float(actions[i][1]), float(actions[i][2]), float(actions[i][3])])
                            vehiclecontrols.append(get_control(actor, actions[i], control))
                        client.apply_batch(vehiclecontrols)
                        world.tick()
                        
                        goals = []
                        action_latent = []
                        for i, actor in enumerate(vehicles_list):
                            tr = actor.get_transform()
                            f = tr.get_forward_vector()
                            v = actor.get_velocity()
                            px, py = tr.location.x - first_pos[i][0], tr.location.y - first_pos[i][1]
                            tx, ty = px + f.x * 5.0, py + f.y * 5.0
                            dx1, dy1 = route[route_target_iter[i]][0], route[route_target_iter[i]][1]
                            if route_target_iter[i] != len(route) - 1:
                                dx2, dy2 = route[route_target_iter[i] + 1][0], route[route_target_iter[i] + 1][1]
                                if i == 0:
                                    print(px, py, dx1, dy1, dx2, dy2)
                                if ((tx - dx1) ** 2 + (ty - dy1) ** 2) > ((tx - dx2) ** 2 + (ty - dy2) ** 2):
                                    route_target_iter[i] += 1
                                    dx1, dy1 = dx2, dy2
                                yaw = tr.rotation.yaw * -0.017453293
                                yawsin = np.sin(yaw)
                                yawcos = np.cos(yaw)
                                dx, dy = rotate(dx1 - px, dy1 - py, yawsin, yawcos)
                                if controller < 3:
                                    d = np.sqrt(dx ** 2 + dy ** 2) / 2.
                                    goals.append([dx / d, dy / d])
                                else:
                                    steer = int(np.round(dy * 10 / dx))
                                    if steer < -10:
                                        steer = -10
                                    elif steer > 10:
                                        steer = 10
                                    if controller == 3:
                                        action_latent.append([latent_task[steer + 10][0], latent_task[steer + 10][1], -1.1])
                                    else:
                                        action_latent.append([latent_task[steer + 10][0], latent_task[steer + 10][1], maxlatent[task]])
                            else:
                                if controller < 3:
                                    goals.append([0.0, 0.0])
                                elif controller == 3:
                                    action_latent.append([-2.0, 1.6, -1.1])
                                else:
                                    action_latent.append([-2.0, 1.6,  maxlatent[task]])

                            if route_current_iter[i] != len(route) - 1:
                                dx1, dy1 = route[route_current_iter[i]][0], route[route_current_iter[i]][1]
                                dx2, dy2 = route[route_current_iter[i] + 1][0], route[route_current_iter[i] + 1][1]
                                lat = abs((dx2 - dx1) * (dy1 - py) - (dx1 - px) * (dy2 - dy1)) / 2.
                                if ((px - dx1) ** 2 + (py - dy1) ** 2) > ((px - dx2) ** 2 + (py - dy2) ** 2):
                                    route_current_iter[i] += 1
                            else:
                                lat = 0.
                            log_pos[i].append([px, py])
                            log_vel[i].append([v.x, v.y])
                            log_distance[i].append(lat)
                            log_routeit[i].append(route_current_iter[i])
                        


                        states = []
                        for i, actor in enumerate(vehicles_list):
                            state = get_state(i, actor)
                            states.append(state)

                    with open(log_name + "task_" + str(task) + "_control_" + str(controller) + "_route_" + str(route_it) + ".json", 'w') as f:
                        json.dump({"pos" : log_pos, "vel" : log_vel, "distance" : log_distance, "routeit" : log_routeit,
                                    "wheel" : log_wheel, "action" : log_action}, f, indent=2)

                    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

                            






                




finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)