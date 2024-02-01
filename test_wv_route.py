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

state_len = 10
action_len = 4
goal_len = 2
traj_len = 50
traj_track_len = 5
latent_len = 3
latent_body_len = 2
latent_preserve = 4
task_num = 11
env_num = 100



route_straight = [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0], [12.0, 0.0], [14.0, 0.0], [16.0, 0.0], [18.0, 0.0], [20.0, 0.0], [22.0, 0.0], [24.0, 0.0], [26.0, 0.0], [28.0, 0.0], [30.0, 0.0], [32.0, 0.0], [34.0, 0.0], [36.0, 0.0], [38.0, 0.0], [40.0, 0.0], [42.0, 0.0], [44.0, 0.0], [46.0, 0.0], [48.0, 0.0], [50.0, 0.0], [52.0, 0.0], [54.0, 0.0], [56.0, 0.0], [58.0, 0.0], [60.0, 0.0], [62.0, 0.0], [64.0, 0.0], [66.0, 0.0], [68.0, 0.0], [70.0, 0.0], [72.0, 0.0], [74.0, 0.0], [76.0, 0.0], [78.0, 0.0], [80.0, 0.0], [82.0, 0.0], [84.0, 0.0], [86.0, 0.0], [88.0, 0.0], [90.0, 0.0], [92.0, 0.0], [94.0, 0.0], [96.0, 0.0], [98.0, 0.0], [100.0, 0.0], [102.0, 0.0], [104.0, 0.0], [106.0, 0.0], [108.0, 0.0], [110.0, 0.0], [112.0, 0.0], [114.0, 0.0], [116.0, 0.0], [118.0, 0.0], [120.0, 0.0], [122.0, 0.0], [124.0, 0.0], [126.0, 0.0], [128.0, 0.0], [130.0, 0.0], [132.0, 0.0], [134.0, 0.0], [136.0, 0.0], [138.0, 0.0], [140.0, 0.0], [142.0, 0.0], [144.0, 0.0], [146.0, 0.0], [148.0, 0.0], [150.0, 0.0], [152.0, 0.0], [154.0, 0.0], [156.0, 0.0], [158.0, 0.0], [160.0, 0.0], [162.0, 0.0], [164.0, 0.0], [166.0, 0.0], [168.0, 0.0], [170.0, 0.0], [172.0, 0.0], [174.0, 0.0], [176.0, 0.0], [178.0, 0.0], [180.0, 0.0]]
route_leftturn = [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0], [12.0, 0.0], [14.0, 0.0], [16.0, 0.0], [18.0, 0.0], [20.0, 0.0], [22.0, 0.0], [24.0, 0.0], [26.0, 0.0], [28.0, 0.0], [30.0, 0.0], [32.0, 0.0], [34.0, 0.0], [36.0, 0.0], [38.0, 0.0], [40.0, 0.0], [42.0, 0.0], [44.0, 0.0], [46.0, 0.0], [48.0, 0.0], [50.0, 0.0], [52.0, 0.0], [54.0, 0.0], [56.0, 0.0], [58.0, 0.0], [60.0, 0.0], [62.0, 0.0], [64.0, 0.0], [66.0, 0.0], [68.0, 0.0], [70.0, 0.0], [72.0, 0.0], [74.0, 0.0], [75.9972590693584, -0.10467191536228049], [77.98235137182199, -0.3484106088432736], [79.93864657089418, -0.7642340017483837], [81.84075959814405, -1.3822680069344768], [83.6533751620733, -2.2275045521698322], [85.31145028892918, -3.345890386174309], [86.72566382075165, -4.760103979098025], [87.84404958311882, -6.4181791542738695], [88.68928605003993, -8.230794754721664], [89.30731997304503, -10.132907808673748], [89.7231432814282, -12.089203025711624], [89.96688188914307, -14.074295338705964], [90.07155371821356, -16.071554412586725], [90.07155363180335, -18.07155441258672], [90.07155354539314, -20.07155441258672], [90.07155345898293, -22.07155441258672], [90.07155337257272, -24.07155441258672], [90.07155328616251, -26.07155441258672], [90.0715531997523, -28.07155441258672], [90.07155311334209, -30.07155441258672], [90.07155302693188, -32.07155441258672], [90.07155294052167, -34.07155441258672], [90.07155285411146, -36.07155441258672], [90.07155276770125, -38.07155441258672], [90.07155268129104, -40.07155441258672], [90.07155259488083, -42.07155441258672], [90.07155250847062, -44.07155441258672], [90.0715524220604, -46.07155441258672], [90.0715523356502, -48.07155441258672], [90.07155224923999, -50.07155441258672], [90.07155216282978, -52.07155441258672], [90.07155207641956, -54.07155441258672], [90.07155199000935, -56.07155441258672], [90.07155190359914, -58.07155441258672], [90.07155181718893, -60.07155441258672], [90.07155173077872, -62.07155441258672], [90.07155164436851, -64.07155441258672], [90.0715515579583, -66.07155441258672], [90.07155147154809, -68.07155441258672], [90.07155138513788, -70.07155441258672], [90.07155129872767, -72.07155441258672], [90.07155121231746, -74.07155441258672], [90.07155112590725, -76.07155441258672], [90.07155103949704, -78.07155441258672], [90.07155095308683, -80.07155441258672], [90.07155086667662, -82.07155441258672], [90.0715507802664, -84.07155441258672], [90.0715506938562, -86.07155441258672], [90.07155060744599, -88.07155441258672], [90.07155052103577, -90.07155441258672], [90.07155043462556, -92.07155441258672], [90.07155034821535, -94.07155441258672], [90.07155026180514, -96.07155441258672]]
route_rightturn = [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0], [12.0, 0.0], [14.0, 0.0], [16.0, 0.0], [18.0, 0.0], [20.0, 0.0], [22.0, 0.0], [24.0, 0.0], [26.0, 0.0], [28.0, 0.0], [30.0, 0.0], [32.0, 0.0], [34.0, 0.0], [36.0, 0.0], [38.0, 0.0], [40.0, 0.0], [42.0, 0.0], [44.0, 0.0], [46.0, 0.0], [48.0, 0.0], [50.0, 0.0], [52.0, 0.0], [54.0, 0.0], [56.0, 0.0], [58.0, 0.0], [60.0, 0.0], [62.0, 0.0], [64.0, 0.0], [66.0, 0.0], [68.0, 0.0], [70.0, 0.0], [72.0, 0.0], [74.0, 0.0], [75.99238939576509, 0.1743114902776158], [77.93298084506526, 0.6581552945192664], [79.74559640899452, 1.5033918397546215], [81.32161789374598, 2.7347148191559665], [82.55294080505516, 4.310736357106862], [83.39817727197627, 6.1233519575546556], [83.88202099237446, 8.063943427759353], [84.0563323965707, 10.056332831055592], [84.05633231016049, 12.05633283105559], [84.05633222375027, 14.056332831055588], [84.05633213734006, 16.056332831055585], [84.05633205092985, 18.056332831055585], [84.05633196451964, 20.056332831055585], [84.05633187810943, 22.056332831055585], [84.05633179169922, 24.056332831055585], [84.05633170528901, 26.056332831055585], [84.0563316188788, 28.056332831055585], [84.05633153246859, 30.056332831055585], [84.05633144605838, 32.056332831055585], [84.05633135964817, 34.056332831055585], [84.05633127323796, 36.056332831055585], [84.05633118682775, 38.056332831055585], [84.05633110041754, 40.056332831055585], [84.05633101400733, 42.056332831055585], [84.05633092759712, 44.056332831055585], [84.0563308411869, 46.056332831055585], [84.0563307547767, 48.056332831055585], [84.05633066836648, 50.056332831055585], [84.05633058195627, 52.056332831055585], [84.05633049554606, 54.056332831055585], [84.05633040913585, 56.056332831055585], [84.05633032272564, 58.056332831055585], [84.05633023631543, 60.056332831055585], [84.05633014990522, 62.056332831055585], [84.05633006349501, 64.05633283105558], [84.0563299770848, 66.05633283105558], [84.05632989067459, 68.05633283105558], [84.05632980426438, 70.05633283105558], [84.05632971785417, 72.05633283105558], [84.05632963144396, 74.05633283105558], [84.05632954503375, 76.05633283105558], [84.05632945862354, 78.05633283105558], [82.05632945862355, 78.05633265823518], [82.05632937221334, 80.05633265823518], [82.05632928580313, 82.05633265823518], [82.05632919939292, 84.05633265823518], [82.05632911298271, 86.05633265823518], [82.0563290265725, 88.05633265823518], [82.05632894016229, 90.05633265823518], [82.05632885375208, 92.05633265823518], [82.05632876734187, 94.05633265823518], [82.05632868093166, 96.05633265823518]]
route_uturn = [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0], [12.0, 0.0], [14.0, 0.0], [16.0, 0.0], [18.0, 0.0], [20.0, 0.0], [22.0, 0.0], [24.0, 0.0], [26.0, 0.0], [28.0, 0.0], [30.0, 0.0], [32.0, 0.0], [34.0, 0.0], [36.0, 0.0], [38.0, 0.0], [40.0, 0.0], [42.0, 0.0], [44.0, 0.0], [46.0, 0.0], [48.0, 0.0], [50.0, 0.0], [52.0, 0.0], [54.0, 0.0], [56.0, 0.0], [58.0, 0.0], [60.0, 0.0], [62.0, 0.0], [64.0, 0.0], [66.0, 0.0], [68.0, 0.0], [70.0, 0.0], [72.0, 0.0], [74.0, 0.0], [75.99878165397118, -0.06979899532405902], [77.99117104973627, -0.2441104856016748], [79.97626335219985, -0.4878491790826679], [81.95164003203837, -0.800718117697765], [83.92125553639556, -1.148014482486897], [85.8845099012757, -1.5296324836071942], [87.84776426615585, -1.9112504847274914], [89.81737977051304, -2.258546849516623], [91.79275645035156, -2.5714157881317203], [93.77784875281515, -2.8151544816127134], [95.77023814858023, -2.989465971890329], [97.76901980255141, -3.0592649672143883], [99.76901980255141, -3.0592649672143883], [101.76901980255141, -3.0592649672143883], [103.76901980255141, -3.0592649672143883], [105.76901980255141, -3.0592649672143883], [107.76901980255141, -3.0592649672143883], [109.76901980255141, -3.0592649672143883], [111.76901980255141, -3.0592649672143883], [113.76901980255141, -3.0592649672143883], [115.76901980255141, -3.0592649672143883], [117.76901980255141, -3.0592649672143883], [119.76901980255141, -3.0592649672143883], [121.76901980255141, -3.0592649672143883], [123.76901980255141, -3.0592649672143883], [125.76901980255141, -3.0592649672143883], [127.76901980255141, -3.0592649672143883], [129.7690198025514, -3.0592649672143883], [131.7690198025514, -3.0592649672143883], [133.7690198025514, -3.0592649672143883], [135.7690198025514, -3.0592649672143883], [137.7690198025514, -3.0592649672143883], [139.7690198025514, -3.0592649672143883], [141.7690198025514, -3.0592649672143883], [143.7690198025514, -3.0592649672143883], [145.7690198025514, -3.0592649672143883], [147.7690198025514, -3.0592649672143883], [149.7690198025514, -3.0592649672143883], [151.7690198025514, -3.0592649672143883], [153.7690198025514, -3.0592649672143883], [155.7690198025514, -3.0592649672143883], [157.7690198025514, -3.0592649672143883], [159.7690198025514, -3.0592649672143883], [161.7690198025514, -3.0592649672143883], [163.7690198025514, -3.0592649672143883], [165.7690198025514, -3.0592649672143883], [167.7690198025514, -3.0592649672143883], [169.7690198025514, -3.0592649672143883], [171.7690198025514, -3.0592649672143883], [173.7690198025514, -3.0592649672143883], [175.7690198025514, -3.0592649672143883], [177.7690198025514, -3.0592649672143883], [179.7690198025514, -3.0592649672143883]]

latent_task = [[1.3, -1.2], #<- -1.0
    [1.4, -1.1], [1.2, -1.1], [1.0, -1.1], [1.1, -1.0], [1.4, -0.8], #<- -0.75
    [0.8, -1.0], [0.8, -0.9], [0.8, -0.8], [1.1, -0.5], [0.9, -0.5], #<- -0.5
    [0.7, -0.5], [0.5, -0.6], [0.7, -0.1], [0.6, 0.0], [0.5, 0.1], #<- -0.25
    [0.4, 0.2], [0.3, 0.2], [0.2, 0.1], [0.1, 0.1], [0.0, 0.5], #<- 0.0
    [-0.1, 0.2], [-0.2, 0.0], [-0.3, 0.1], [-0.4, 0.0], [-0.5, 0.0], #<- 0.25
    [-0.7, 0.1], [-0.7, -0.2], [-0.9, -0.2], [-0.9, -0.4], [-0.9, -0.6], #<- 0.5
    [-1.0, -0.6], [-1.2, -0.6], [-1.1, -0.8], [-1.4, -0.7], [-1.4, -0.8], #<- 0.75
    [-0.9, -1.2], [-1.3, -1.0], [-1.6, -0.9], [-1.8, -0.9], [-1.7, 1.0]] #<- 1.0

maxlatent = [3.7, -1.8, 10.1, 11.9, -3.7, 11.3, -7.3, -0.396, -3.668, 1.668, 5.0, -1.01]
latentcand = [0.284, -0.2, 1.028]

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos
  
log_name = "test_log/Train3_6_2/route_test2/"

def get_state(i, actor):
    tr = actor.get_transform()
    v = actor.get_velocity()
    a = actor.get_acceleration()
    yaw = (tr.rotation.yaw + yaw_diff) * -0.017453293
    f = tr.get_forward_vector()
    px, py = rotate(tr.location.x - first_traj_pos[i][0], tr.location.y - first_traj_pos[i][1], first_traj_yaw[i][0], first_traj_yaw[i][1])
    vx, vy = rotate(v.x, v.y, first_traj_yaw[i][0], first_traj_yaw[i][1])
    ax, ay = rotate(a.x, a.y, first_traj_yaw[i][0], first_traj_yaw[i][1])
    fx, fy = rotate(f.x, f.y, first_traj_yaw[i][0], first_traj_yaw[i][1])
    return [px, py, vx, vy, ax, ay, fx, fy, tr.rotation.roll, tr.rotation.pitch]



def get_control(actor, action, steer):
    control = carla.VehicleControl()
    if controller < 3:
        action = np.tanh(action + 0.25)
    if ratio_fl < 0.:
        action[0] = (action[1] + action[2] + action[3]) / 3
    if ratio_fr < 0.:
        action[1] = (action[0] + action[2] + action[3]) / 3
    if ratio_rl < 0.:
        action[2] = (action[0] + action[1] + action[3]) / 3
    if ratio_rr < 0.:
        action[3] = (action[1] + action[1] + action[2]) / 3
    control.fl=float(action[0] * ratio_fl)
    control.fr=float(action[1] * ratio_fr)
    control.bl=float(action[2] * ratio_rl)
    control.br=float(action[3] * ratio_rr)
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
        learner_saver.restore(sess, "train_log/Train3_6_2/log_2023-11-14-18-00-49_learner_440.ckpt")
        explorer = [Skill_Explorer(state_len,  action_len, goal_len, name=str(i)) for i in range(task_num)]
        explorer_savers = [tf.train.Saver(max_to_keep=0, var_list=explorer[i].trainable_dict) for i in range(task_num)]
        explorer_savers[0].restore(sess, "train_log/Train3_2/log_2023-10-13-18-24-03_explorer0_200.ckpt")
        explorer_savers[1].restore(sess, "train_log/Train3_2/log_2023-10-13-18-24-03_explorer1_200.ckpt")
        explorer_savers[2].restore(sess, "train_log/Train3_2/log_2023-10-13-18-24-03_explorer2_200.ckpt")

        latent_for_task = []
        for task in range(0, 7):
            ratio_fl = 100.
            ratio_fr = 100.
            ratio_rl = 100.
            ratio_rr = 100.
            yaw_diff = 0.
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
                yaw_diff = -15.
            elif task == 8:
                yaw_diff = 15.
            elif task == 9:
                yaw_diff = -30.
            elif task == 10:
                yaw_diff = 30.

            env_num = 16
            for route_it, route in enumerate([route_straight, route_leftturn, route_rightturn, route_uturn]): 
                for controller in range(4):
                    log_pos = [[] for _ in range(env_num)]
                    log_vel = [[] for _ in range(env_num)]
                    log_distance = [[] for _ in range(env_num)]
                    log_routeit = [[] for _ in range(env_num)]
                    log_wheel = [[] for _ in range(env_num)]
                    log_action = [[] for _ in range(env_num)]
                    log_vt = [[] for _ in range(env_num)]

                    vehicles_list = []
                    for x, y in itertools.product(range(4), range(4)):                                                                                                                                                                                                                           
                        spawn_point = carla.Transform(carla.Location(x * 400. - 1000., y * 400. - 1000., 3.0), carla.Rotation(0, 0, 0))
                        actor = world.spawn_actor(blueprints[task], spawn_point)
                        vehicles_list.append(actor)
                    for step in range(30):
                        world.tick()

                    first_pos = []
                    first_traj_pos = []
                    first_traj_yaw = []
                    states = []
                    action_latent = []
                    action_steer = []
                    goals = []
                    for i, actor in enumerate(vehicles_list):
                        tr = actor.get_transform()
                        first_pos.append([tr.location.x, tr.location.y])
                        first_traj_pos.append([tr.location.x, tr.location.y])
                        first_traj_yaw.append([np.sin((tr.rotation.yaw + yaw_diff) * -0.017453293), np.cos((tr.rotation.yaw + yaw_diff) * -0.017453293)])
                        states.append(get_state(i, actor))
                        if controller < 3:
                            goals.append([2.0, 0.0])
                            action_steer.append(0.)
                        elif controller == 3:
                            action_latent.append([0.0, 0.5, maxlatent[task]])
                            action_steer.append(0.)
                        else:
                            action_latent.append([0.0, 0.5, latentcand[controller - 4]])
                            action_steer.append(0.)

                    route_target_iter = [0 for _ in range(env_num)]
                    route_current_iter = [0 for _ in range(env_num)]



                    for step in range(2000):
                        if controller < 3:
                            actions = explorer[controller].get_action(states, goals, True)
                        else:
                            actions = learner.get_action(states, action_latent, True)
                        vehiclecontrols = []
                        for i, actor in enumerate(vehicles_list):
                            control = actor.get_control()
                            log_wheel[i].append([control.fl, control.fr, control.bl, control.br])
                            log_action[i].append([float(actions[i][0]), float(actions[i][1]), float(actions[i][2]), float(actions[i][3])])
                            vehiclecontrols.append(get_control(actor, actions[i], action_steer[i]))
                        client.apply_batch(vehiclecontrols)
                        world.tick()
                        
                        if step % traj_track_len == 0:
                            first_traj_pos = []
                            first_traj_yaw = []
                            goals = []
                            action_latent = []
                            action_steer = []
                            for i, actor in enumerate(vehicles_list):

                                tr = actor.get_transform()
                                f = tr.get_forward_vector()
                                v = actor.get_velocity()
                                px, py = tr.location.x - first_pos[i][0], tr.location.y - first_pos[i][1]
                                tx, ty = px + f.x * 3.0, py + f.y * 3.0
                                dx1, dy1 = route[route_target_iter[i]][0], route[route_target_iter[i]][1]
                                yaw = (tr.rotation.yaw + yaw_diff) * -0.017453293
                                yawsin = np.sin(yaw)
                                yawcos = np.cos(yaw)
                                if route_target_iter[i] != len(route) - 1:
                                    dx2, dy2 = route[route_target_iter[i] + 1][0], route[route_target_iter[i] + 1][1]
                                    while ((tx - dx1) ** 2 + (ty - dy1) ** 2) > ((tx - dx2) ** 2 + (ty - dy2) ** 2):
                                        route_target_iter[i] += 1
                                        if route_target_iter[i] == len(route) - 1:
                                            break
                                        dx1, dy1 = dx2, dy2
                                        dx2, dy2 = route[route_target_iter[i] + 1][0], route[route_target_iter[i] + 1][1]
                                    dx, dy = rotate(dx1 - px, dy1 - py, yawsin, yawcos)
                                    if controller < 3:
                                        d = np.sqrt(dx ** 2 + dy ** 2)
                                        goals.append([dx * 2 / d, dy * 2 / d])
                                        action_steer.append(0.)
                                        if i == 0:
                                            print(px, py, dx, dy, goals[0])
                                    else:
                                        #steer = int(np.round(dy * 20 / dx))
                                        steer = int(np.round(dy * 20))
                                        if steer < -20:
                                            steer = -20
                                        elif steer > 20:
                                            steer = 20
                                        if i == 0:
                                            print(px, py, dx, dy, steer)
                                        if controller == 3:
                                            action_latent.append([latent_task[steer + 20][0], latent_task[steer + 20][1], maxlatent[task]])
                                        else:
                                            action_latent.append([latent_task[steer + 20][0], latent_task[steer + 20][1], latentcand[controller - 4]])
                                        action_steer.append(steer)
                                else:
                                    if controller < 3:
                                        goals.append([0.0, 0.0])
                                    elif controller == 3:
                                        action_latent.append([-2.0, -2.0, maxlatent[task]])
                                    else:
                                        action_latent.append([-2.0, -2.0, latentcand[controller - 4]])
                                    action_steer.append(0.)
                                first_traj_pos.append([tr.location.x, tr.location.y])
                                first_traj_yaw.append([yawsin, yawcos])

                        for i, actor in enumerate(vehicles_list):
                            tr = actor.get_transform()
                            v = actor.get_velocity()
                            px, py = tr.location.x - first_pos[i][0], tr.location.y - first_pos[i][1]
                            if route_current_iter[i] != len(route) - 1:
                                dx1, dy1 = route[route_current_iter[i]][0], route[route_current_iter[i]][1]
                                dx2, dy2 = route[route_current_iter[i] + 1][0], route[route_current_iter[i] + 1][1]
                                lat = ((dx2 - dx1) * (dy1 - py) - (dx1 - px) * (dy2 - dy1)) / 2.
                                if ((px - dx1) ** 2 + (py - dy1) ** 2) > ((px - dx2) ** 2 + (py - dy2) ** 2):
                                    route_current_iter[i] += 1
                            else:
                                lat = 0.
                            log_pos[i].append([px, py])
                            log_vel[i].append([v.x, v.y])
                            log_distance[i].append(lat)
                            log_routeit[i].append(route_current_iter[i])
                            if controller < 3:
                                log_vt[i].append([goals[i][0], goals[i][1]])
                            else:
                                log_vt[i].append([action_latent[i][0], action_latent[i][1]])
                        


                        states = []
                        for i, actor in enumerate(vehicles_list):
                            state = get_state(i, actor)
                            states.append(state)

                    with open(log_name + "task_" + str(task) + "_control_" + str(controller) + "_route_" + str(route_it) + ".json", 'w') as f:
                        json.dump({"pos" : log_pos, "vel" : log_vel, "distance" : log_distance, "routeit" : log_routeit,
                                    "wheel" : log_wheel, "action" : log_action, "task" : log_vt}, f, indent=2)
                    print(task, controller, route_it)
                    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

                            






                




finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)