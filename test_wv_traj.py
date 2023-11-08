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
traj_track_len = 10
latent_len = 3
latent_body_len = 2
latent_preserve = 4
task_num = 11
env_num = 100



latent_task = [[-2.4, 2.8], #<- -1.0
    [-2.4, 3.4], [-2.2, 2.4], [-2.2, 3.6], [-2.0, 3.6], [-1.8, 2.2], #<- -0.5
    [-1.6, 2.0], [-1.2, 2.4], [-0.6, 2.6], [0.2, 2.4], [2.0, 2.4], #<- 0.0
    [1.0, 0.2], [2.6, -1.0], [1.2, -2.0], [1.2, -3.0], [-0.8, -2.0], #<-0.5
    [0.2, -3.4], [-0.4, -3.0], [-0.2, -3.6], [-0.4, -3.6], [-0.6, -3.4]] #<- 1.0
    
maxlatent = [0.96, 2.4, 0.12, -0.6, 4.52, -1.72, 3.08, -0.64, 2.24]
route = [[141.19, -204.39999982717958], [139.19, -204.39999965435916], [137.19, -204.39999948153874], [135.19, -204.39999930871832], [133.19, -204.3999991358979], [131.19, -204.39999896307748], [129.19, -204.39999879025706], [127.19000000000001, -204.39999861743664], [125.19000000000003, -204.39999844461622], [123.19000000000004, -204.3999982717958], [121.19000000000005, -204.39999809897537], [119.19000000000007, -204.39999792615495], [117.19000000000008, -204.39999775333453], [115.1900000000001, -204.3999975805141], [113.19000000000011, -204.3999974076937], [111.19000000000013, -204.39999723487327], [109.19000000000014, -204.39999706205285], [107.19000000000015, -204.39999688923243], [105.19000000000017, -204.399996716412], [103.19000000000018, -204.39999654359158], [101.1900000000002, -204.39999637077116], [99.19000000000021, -204.39999619795074], [97.19000000000023, -204.39999602513032], [95.19000000000024, -204.3999958523099], [93.19000000000025, -204.39999567948948], [91.19000000000027, -204.39999550666906], [89.19000000000028, -204.39999533384864], [87.1900000000003, -204.39999516102822], [85.19000000000031, -204.3999949882078], [83.19000000000032, -204.39999481538737], [81.19000000000034, -204.39999464256695], [79.19000000000035, -204.39999446974653], [77.19000000000037, -204.3999942969261], [75.19000000000038, -204.3999941241057], [73.1900000000004, -204.39999395128527], [71.19000000000041, -204.39999377846485], [69.19000000000042, -204.39999360564443], [67.19000000000044, -204.399993432824], [65.19000000000045, -204.39999326000358], [63.19000000000046, -204.39999308718316], [61.19000000000047, -204.39999291436274], [59.190000000000474, -204.39999274154232], [57.19000000000048, -204.3999925687219], [55.19000000000049, -204.39999239590148], [53.190000000000495, -204.39999222308106], [51.1900000000005, -204.39999205026064], [49.19000000000051, -204.39999187744021], [47.190000000000516, -204.3999917046198], [45.19000000000052, -204.39999153179937], [43.19000000000053, -204.39999135897895], [41.19000000000054, -204.39999118615853], [39.190000000000545, -204.3999910133381], [37.19000000000055, -204.3999908405177], [35.19000000000056, -204.39999066769727], [33.190000000000566, -204.39999049487685], [31.190000000000573, -204.39999032205642], [29.19000000000058, -204.399990149236], [27.190000000000587, -204.39998997641558], [25.190000000000595, -204.39998980359516], [23.1900000000006, -204.39998963077474], [21.19000000000061, -204.39998945795432], [19.190000000000616, -204.3999892851339], [17.190000000000623, -204.39998911231348], [15.19000000000063, -204.39998893949306], [13.190000000000637, -204.39998876667264], [11.190000000000644, -204.3999885938522], [9.190000000000651, -204.3999884210318], [7.1900000000006585, -204.39998824821137], [5.190000000000666, -204.39998807539095], [3.190000000000673, -204.39998790257053], [1.1900000000006807, -204.3999877297501], [-0.8099999999993117, -204.3999875569297], [-2.809999999999304, -204.39998738410927], [-4.809999999999297, -204.39998721128885], [-6.80999999999929, -204.39998703846842], [-8.809999999999283, -204.399986865648], [-10.809999999999276, -204.39998669282758], [-12.809999999999269, -204.39998652000716], [-14.809999999999262, -204.39998634718674], [-16.809999999999253, -204.39998617436632], [-18.809999999999246, -204.3999860015459], [-20.80999999999924, -204.39998582872548], [-22.80999999999923, -204.39998565590506], [-24.809999999999224, -204.39998548308463], [-26.809999999999217, -204.3999853102642], [-28.80999999999921, -204.3999851374438], [-30.809999999999203, -204.39998496462337], [-32.80999999999919, -204.39998479180295], [-34.809999999999185, -204.39998461898253], [-36.80999999999918, -204.3999844461621], [-38.80999999999917, -204.3999842733417], [-40.809999999999164, -204.39998410052127], [-42.80999999999916, -204.39998392770084], [-44.80999999999915, -204.39998375488042], [-46.80999999999914, -204.39998358206], [-48.809999999999135, -204.39998340923958], [-50.80999999999913, -204.39998323641916], [-52.80999999999912, -204.39998306359874], [-54.809999999999114, -204.39998289077832], [-56.80999999999911, -204.3999827179579], [-58.8099999999991, -204.39998254513748], [-60.80999999999909, -204.39998237231706], [-62.809999999999086, -204.39998219949663], [-64.80999999999908, -204.3999820266762], [-66.80999999999906, -204.3999818538558], [-68.80999999999905, -204.39998168103537], [-70.80999999999904, -204.39998150821495], [-72.80999999999902, -204.39998133539453], [-74.80999999999901, -204.3999811625741], [-76.809999999999, -204.3999809897537], [-78.80999999999898, -204.39998081693327], [-80.80999999999896, -204.39998064411284], [-82.80999999999895, -204.39998047129242], [-84.80999999999894, -204.399980298472], [-86.80999999999892, -204.39998012565158], [-88.80999999999891, -204.39997995283116], [-90.8099999999989, -204.39997978001074], [-92.80999999999888, -204.39997960719032], [-94.80999999999887, -204.3999794343699], [-96.80999999999885, -204.39997926154948], [-98.80999999999884, -204.39997908872905], [-100.80999999999882, -204.39997891590863], [-102.80999999999881, -204.3999787430882], [-104.8099999999988, -204.3999785702678], [-106.80999999999878, -204.39997839744737], [-108.80999999999877, -204.39997822462695], [-110.80999999999875, -204.39997805180653], [-112.80999999999874, -204.3999778789861], [-114.80999999999872, -204.39997770616569], [-116.80999999999871, -204.39997753334526], [-118.8099999999987, -204.39997736052484], [-120.80999999999868, -204.39997718770442], [-122.80999999999867, -204.399977014884], [-124.80999999999865, -204.39997684206358], [-126.80999999999864, -204.39997666924316], [-128.80999999999864, -204.39997649642274], [-130.80999999999864, -204.39997632360232], [-132.80999999999864, -204.3999761507819], [-134.80999999999864, -204.39997597796147], [-136.80999999999864, -204.39997580514105], [-138.80999999999864, -204.39997563232063], [-140.80999999999864, -204.3999754595002], [-142.80999999999864, -204.3999752866798], [-144.80999999999864, -204.39997511385937], [-146.80999999999864, -204.39997494103895], [-148.80999999999864, -204.39997476821853], [-150.80999999999864, -204.3999745953981], [-152.80999999999864, -204.39997442257769], [-154.80999999999864, -204.39997424975726], [-156.80999999999864, -204.39997407693684], [-158.80999999999864, -204.39997390411642], [-160.80999999999864, -204.399973731296], [-162.80999999999864, -204.39997355847558], [-164.80999999999864, -204.39997338565516], [-166.80999999999864, -204.39997321283474], [-168.80999999999864, -204.39997304001432], [-170.80999999999864, -204.3999728671939], [-172.80999999999864, -204.39997269437347], [-174.80999999999864, -204.39997252155305], [-176.80999999999864, -204.39997234873263], [-178.80999999999864, -204.3999721759122], [-180.80999999999864, -204.3999720030918], [-182.80999999999864, -204.39997183027137], [-184.80999999999864, -204.39997165745095], [-186.80999999999864, -204.39997148463053], [-188.80999999999864, -204.3999713118101], [-190.80999999999864, -204.39997113898968], [-192.80999999999864, -204.39997096616926], [-194.80999999999864, -204.39997079334884], [-196.80999999999864, -204.39997062052842], [-198.80999999999864, -204.399970447708], [-200.80999999999864, -204.39997027488758], [-202.80999999999864, -204.39997010206716], [-204.80999999999864, -204.39996992924674], [-206.80999999999864, -204.39996975642632], [-208.80999999999864, -204.3999695836059], [-210.80999999999864, -204.39996941078547], [-212.80999999999864, -204.39996923796505], [-214.80999999999864, -204.39996906514463], [-216.80999999999864, -204.3999688923242], [-218.80999999999864, -204.3999687195038], [-220.80999999999864, -204.39996854668337], [-222.80999999999864, -204.39996837386295], [-224.80999999999864, -204.39996820104253], [-226.80999999999864, -204.3999680282221], [-228.80999999999864, -204.39996785540168], [-230.80999999999864, -204.39996768258126], [-232.80999999999864, -204.39996750976084], [-234.80999999999864, -204.39996733694042], [-236.80999999999864, -204.39996716412]]

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos
  
log_name = "test_log/Train3_2/traj_test/"

def get_state(i, actor):
    tr = actor.get_transform()
    v = actor.get_velocity()
    a = actor.get_acceleration()
    yaw = tr.rotation.yaw * -0.017453293
    f = tr.get_forward_vector()
    px, py = rotate(tr.location.x - first_traj_pos[i][0], tr.location.y - first_traj_pos[i][1], first_traj_yaw[i][0], first_traj_yaw[i][1])
    vx, vy = rotate(v.x, v.y, first_traj_yaw[i][0], first_traj_yaw[i][1])
    ax, ay = rotate(a.x, a.y, first_traj_yaw[i][0], first_traj_yaw[i][1])
    fx, fy = rotate(f.x, f.y, first_traj_yaw[i][0], first_traj_yaw[i][1])
    return [px, py, vx, vy, ax, ay, fx, fy, tr.rotation.roll, tr.rotation.pitch]



def get_control(actor, action):
    control = carla.VehicleControl()
    action = np.tanh(action)
    control.fl=float(action[0] * ratio_fl)
    control.fr=float(action[1] * ratio_fr)
    control.bl=float(action[2] * ratio_rl)
    control.br=float(action[3] * ratio_rr)
    control.gear = 1
    control.manual_gear_shift = True
    control.hand_brake = False
    control.reverse = False
    return control

try:
    with sess.as_default():
        world = client.get_world()
        world_map = world.get_map()
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
        learner_saver.restore(sess, "train_log/Train3_2/log_2023-10-14-01-18-22_learner_300.ckpt")
        explorer = [Skill_Explorer(state_len,  action_len, goal_len, name=str(i)) for i in range(task_num)]
        explorer_savers = [tf.train.Saver(max_to_keep=0, var_list=explorer[i].trainable_dict) for i in range(task_num)]
        explorer_savers[0].restore(sess, "train_log/Train3_2/log_2023-10-14-01-18-22_explorer0_300.ckpt")
        explorer_savers[1].restore(sess, "train_log/Train3_2/log_2023-10-14-01-18-22_explorer1_300.ckpt")
        explorer_savers[2].restore(sess, "train_log/Train3_2/log_2023-10-14-01-18-22_explorer2_300.ckpt")


        latent_for_task = []
        ratio_fl = 50.
        ratio_fr = 50.
        ratio_rl = 50.
        ratio_rr = 50.

        env_num = 16
        log_pos = [[] for _ in range(env_num)]
        log_vel = [[] for _ in range(env_num)]
        log_distance = [[] for _ in range(env_num)]
        log_routeit = [[] for _ in range(env_num)]
        log_wheel = [[] for _ in range(env_num)]
        log_action = [[] for _ in range(env_num)]
        log_latent = [[] for _ in range(env_num)]

        for env in range(env_num):
            ratio_fl = 50.
            ratio_fr = 50.
            ratio_rl = 50.
            ratio_rr = 50.
            vehicles_list = []
            actor = world.spawn_actor(blueprints[0], carla.Transform(carla.Location(143.19, -204.40, 5.0), carla.Rotation(0, 185, 0)))
            vehicles_list.append(actor)
            step = 0
            tr = actor.get_transform()
            first_traj_pos = [[tr.location.x, tr.location.y]]
            first_traj_yaw = [[np.sin(tr.rotation.yaw * -0.017453293), np.cos(tr.rotation.yaw * -0.017453293)]]
            state = get_state(0, actor)
            va = -1.1
            action_latent = [[1.5, -0.8, va]]

            route_target_iter = [0]
            route_current_iter = [0]

            for step in range(30):
                world.tick()
            for step in range(400):
                if step > 150:
                    ratio_fl = -25.
                actions = learner.get_action([state], action_latent, True)
                control = actor.get_control()
                log_wheel[env].append([control.fl, control.fr, control.bl, control.br])
                log_action[env].append([float(actions[0][0]), float(actions[0][1]), float(actions[0][2]), float(actions[0][3])])
                actor.apply_control(get_control(actor, actions[0], control))
                world.tick()

                first_traj_pos = []
                first_traj_yaw = []
                action_latent = []
                for i, actor in enumerate(vehicles_list):
                    tr = actor.get_transform()
                    f = tr.get_forward_vector()
                    px, py = tr.location.x, tr.location.y
                    tx, ty = px + f.x * 3.0, py + f.y * 3.0
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

                        steer = int(np.round(dy * 10 / dx))
                        if steer < -10:
                            steer = -10
                        elif steer > 10:
                            steer = 10
                        action_latent.append([latent_task[steer + 10][0], latent_task[steer + 10][1], va])
                        print(action_latent)
                    else:
                        action_latent.append([-2.0, 1.6, va])
                    first_traj_pos.append([tr.location.x, tr.location.y])
                    first_traj_yaw.append([np.sin(tr.rotation.yaw * -0.017453293), np.cos(tr.rotation.yaw * -0.017453293)])

                for i, actor in enumerate(vehicles_list):
                    tr = actor.get_transform()
                    v = actor.get_velocity()
                    px, py = tr.location.x, tr.location.y
                    if route_current_iter[i] != len(route) - 1:
                        dx1, dy1 = route[route_current_iter[i]][0], route[route_current_iter[i]][1]
                        dx2, dy2 = route[route_current_iter[i] + 1][0], route[route_current_iter[i] + 1][1]
                        lat = abs((dx2 - dx1) * (dy1 - py) - (dx1 - px) * (dy2 - dy1)) / 2.
                        if ((px - dx1) ** 2 + (py - dy1) ** 2) > ((px - dx2) ** 2 + (py - dy2) ** 2):
                            route_current_iter[i] += 1
                    else:
                        lat = 0.
                    if lat > 0.5:
                        if va < 2.:
                            va += 0.05
                        ratio_fl += 5.
                        ratio_rl += 5.
                    elif lat > 1.:
                        if va < 2.:
                            va += 0.1
                        ratio_fl += 10.
                        ratio_rl += 10.
                    elif lat < -0.5:
                        if va > -2.:
                            va -= 0.05
                        ratio_fl -= 5.
                        ratio_rl -= 5.
                    log_pos[env].append([px, py])
                    log_vel[env].append([v.x, v.y])
                    log_distance[env].append(lat)
                    log_routeit[env].append(route_current_iter[i])
                    log_latent[env].append(va)
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])


        with open(log_name + "task_1.json", 'w') as f:
            json.dump({"pos" : log_pos, "vel" : log_vel, "distance" : log_distance, "routeit" : log_routeit,
                        "wheel" : log_wheel, "action" : log_action, "latent" : log_latent}, f, indent=2)


                        






                




finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)