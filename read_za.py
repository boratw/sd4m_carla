import numpy as np
import json
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import itertools

matplotlib.rc('font', size=16)
matplotlib.rc('axes', titlesize=16, labelsize=16)

log_dir = 'test_log/Train3_6_2/'
t_score_m = []
t_score_s = []
t_vel_m = []
t_vel_s = []
t_lat_m = []
t_lat_s = []
for task in range(3):
    filename = "task_" + str(task) + "_score.txt"

    za = []
    score_m = []
    score_s = []
    vel_m = []
    vel_s = []
    lat_m = []
    lat_s = []
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    ts1 = []
    ts2 = []
    ts3 = []
    ts4 = []
    with open(log_dir + filename) as f:
        lines = f.readlines()
        for line in lines:
            sep = line.split("\t")
            if len(sep) == 13:
                za.append(float(sep[0]) / 10.)
                d = float(sep[1])
                w = abs(float(sep[3]))
                score_m.append((d if d > 0. else 0.) - w * 2.)
                score_s.append(np.sqrt(float(sep[2])) * 10 +  np.sqrt(float(sep[4])) * 20)
                vel_m.append(d / 75 if d > 0. else 0.)
                vel_s.append(np.sqrt(float(sep[2])) / 7.5)
                lat_m.append(w / 150)
                lat_s.append( np.sqrt(float(sep[4])) / 15)
                t1.append(float(sep[5]) / 200.)
                t2.append(float(sep[7]) / 200.)
                t3.append(float(sep[9]) / 200.)
                t4.append(float(sep[11]) / 200.)
                w1 = float(sep[5])
                w21 = float(sep[6]) + w1 ** 2 / 4
                w21 = w21 / 64 - w1 ** 2 / 4
                ts1.append(w21)
                w1 = float(sep[7])
                w21 = float(sep[8]) + w1 ** 2 / 4
                w21 = w21 / 64 - w1 ** 2 / 4
                ts2.append(w21)
                w1 = float(sep[9])
                w21 = float(sep[10]) + w1 ** 2 / 4
                w21 = w21 / 64 - w1 ** 2 / 4
                ts3.append(w21)
                w1 = float(sep[11])
                w21 = float(sep[12]) + w1 ** 2 / 4
                w21 = w21 / 64 - w1 ** 2 / 4
                ts4.append(w21 )
    for name, lim, m, s in zip(['score', 'velocity (m/s)', 'error (m)'], [[ -250, 250], [1, 3], [0, 2]], [score_m, vel_m, lat_m], [score_s, vel_s, lat_s]):
        plt.figure()
        plt.xlabel('za')
        plt.ylabel(name)
        plt.ylim(lim)
        m = np.array(m)
        s = np.array(s)
        m[1:-1] = (m[:-2] + m[1:-1] + m[2:]) / 3
        s[1:-1] = (s[:-2] + s[1:-1] + s[2:]) / 3
        plt.fill_between(za, m - s,  m + s,
            alpha=0.25, facecolor='red', antialiased=True)
        plt.plot(za, m,  'r-', label=name)
        #plt.legend()
        plt.savefig(log_dir + filename[:-5] + "_" + name[:-4] +".png", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.xlabel('za')
    plt.ylabel('wheel')
    plt.ylim([-0.75, 0.75])
    t1 = np.array(t1)
    t2 = np.array(t2)
    t3 = np.array(t3)
    t4 = np.array(t4)
    ts1 = np.array(ts1)
    ts1[1:-1] = (ts1[:-2] + ts1[1:-1] + ts1[2:]) / 3
    ts2 = np.array(ts2)
    ts2[1:-1] = (ts2[:-2] + ts2[1:-1] + ts2[2:]) / 3
    ts3 = np.array(ts3)
    ts3[1:-1] = (ts3[:-2] + ts3[1:-1] + ts3[2:]) / 3
    ts4 = np.array(ts4)
    ts4[1:-1] = (ts4[:-2] + ts4[1:-1] + ts4[2:]) / 3
    plt.plot(za, t1, 'r-', label='FL')
    plt.plot(za, t2, 'm-', label='FR')
    plt.plot(za, t3, 'g-', label='RL')
    plt.plot(za, t4, 'b-', label='RR')
    plt.fill_between(za, t1 - ts1,  t1 + ts1, alpha=0.25, facecolor='red', antialiased=True)
    plt.fill_between(za, t2 - ts2,  t2 + ts2, alpha=0.25, facecolor='magenta', antialiased=True)
    plt.fill_between(za, t3 - ts3,  t3 + ts3, alpha=0.25, facecolor='green', antialiased=True)
    plt.fill_between(za, t4 - ts4,  t4 + ts4, alpha=0.25, facecolor='blue', antialiased=True)
    plt.legend()
    plt.savefig(log_dir + filename[:-5] +"_wheel.png", dpi=300, bbox_inches="tight")

    t_score_m.append(np.array(score_m))
    t_score_s.append(np.array(score_s))
    t_vel_m.append(np.array(vel_m))
    t_vel_s.append(np.array(vel_s))
    t_lat_m.append(np.array(lat_m))
    t_lat_s.append(np.array(lat_s))
'''
l = len(za)
za = za[l // 4 : l * 3 // 4]
score_m0 = score_m0[l // 4 : l * 3 // 4]
score_m1 = score_m1[l // 4 : l * 3 // 4]
score_m2 = score_m2[l // 4 : l * 3 // 4]
score_s0 = score_s0[l // 4 : l * 3 // 4] / 10
score_s1 = score_s1[l // 4 : l * 3 // 4]
score_s2 = score_s2[l // 4 : l * 3 // 4]
'''
for name, lim, m, s in zip(['score', 'velocity', 'error'], [[ -250, 250], [1, 3], [0, 2]], [t_score_m, t_vel_m, t_lat_m], [t_score_s, t_vel_s, t_lat_s]):
    plt.figure()
    plt.xlabel('za')
    plt.ylabel(name)
    plt.ylim(lim)
    plt.fill_between(za, m[0] - s[0],  m[0] + s[0],
        alpha=0.25, facecolor='red', antialiased=True)
    plt.fill_between(za,  m[1] - s[1],  m[1] + s[1],
        alpha=0.25, facecolor='green', antialiased=True)
    plt.fill_between(za,  m[2] - s[2],  m[2] + s[2],
        alpha=0.25, facecolor='blue', antialiased=True)
    plt.plot(za, m[0],  'r-', label='m1')
    plt.plot(za, m[1],  'g-', label='m2')
    plt.plot(za, m[2],  'b-', label='m3')
    plt.legend()
    plt.savefig(log_dir + filename[:-5] + "_" + name +"_total.png", dpi=300, bbox_inches="tight")