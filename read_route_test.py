import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import os
import itertools
import math


log_dir = 'test_log/Train3_5/route_test2/'
output_dir = 'test_log/Train3_5/route_test2_log/'
#filenames = os.listdir(log_dir)
#for filename in filenames:
#    if filename[-4:] == 'json':
for task in range(4):
    for route in range(4):
        start_index = 10
        end_index= 0
        log_lat = []
        log_vel = []
        log_action0 = []
        log_action1 = []
        log_action2 = []
        log_action3 = []
        for control in [1, 2]:
            filename = "task_" + str(task) + "_control_" + str(control) + "_route_" + str(route) + ".json"
            with open(log_dir + filename) as f:
                json_object = json.load(f)
            
            traj_map = np.full((4096, 8192, 3), 0, np.uint8)
                #cv2.polylines(traj_map, np.array([np.array(route) * 128. + np.array((-2048, 2048))], np.int32), False, (0, 0, 0))
            for i, r in enumerate(json_object['pos']):
                overlay = np.full((4096, 8192, 3), 0, np.uint8)
                cv2.polylines(overlay, np.array([np.array(r) * 64. + np.array((-64 * 64., 0.))+ np.array((4096, 2048))], np.int32), False, (255, 255, 255), 8)
                traj_map = cv2.addWeighted(overlay, 0.05, traj_map, 1., 0)
            cv2.imwrite(output_dir + filename[:-5] + ".png", traj_map)



            lat = []
            vel = []
            action0 = []
            action1 = []
            action2 = []
            action3 = []
            obj_distance = json_object['distance']
            obj_vel = json_object['vel']
            obj_action = json_object['action']
            lastindex = []
            for i, r in enumerate(json_object['routeit']):
                readed_it = 0
                for j, t in enumerate(r):
                    if readed_it != t:
                        while t >= len(lat):
                            lat.append([])
                            vel.append([])
                            action0.append([])
                            action1.append([])
                            action2.append([])
                            action3.append([])
                        lat[t].append(obj_distance[i][j])
                        vel[t].append(math.sqrt(obj_vel[i][j][0] ** 2 + obj_vel[i][j][1] ** 2))
                        action0[t].append(obj_action[i][j][0])
                        action1[t].append(obj_action[i][j][1])
                        action2[t].append(obj_action[i][j][2])
                        action3[t].append(obj_action[i][j][3])
                        readed_it = t
                lastindex.append(readed_it)
            if end_index < np.min(lastindex):
                end_index = np.min(lastindex)

            log_lat.append(lat[1:])
            log_vel.append(vel[1:])
            log_action0.append(action0[1:])
            log_action1.append(action1[1:])
            log_action2.append(action2[1:])
            log_action3.append(action3[1:])

            print(filename)
            with open(output_dir + filename[:-5] + ".txt", "wt") as f:
                for i in range(1, len(lat)):
                    f.write(str(np.mean(lat[i])) + "\t" + str(np.std(lat[i])) + "\t" 
                    + str(np.mean(vel[i])) + "\t" + str(np.std(vel[i])) + "\t" 
                    + str(np.mean(action0[i])) + "\t" + str(np.std(action0[i])) + "\t" 
                    + str(np.mean(action1[i])) + "\t" + str(np.std(action1[i])) + "\t" 
                    + str(np.mean(action2[i])) + "\t" + str(np.std(action2[i])) + "\t" 
                    + str(np.mean(action3[i])) + "\t" + str(np.std(action3[i]))+ "\n")

            plt.figure()
            step = np.arange(10, 70)
            plt.xlabel('step')
            plt.ylabel('wheel')
            plt.ylim([-1.0, 1.0])
            plt.rc('font', size=12)

            for f, c, l, action in zip(['red', 'magenta', 'green', 'blue'], ['ro-', 'm^-', 'g*-', 'bs-'], 
                ['FL', 'FR', 'RL', 'RR'], [action0, action1, action2, action3]):
                action_mean = []
                action_var = []
                for i in range(10, 70):
                    action_mean.append(np.mean(action[i]))
                    action_var.append(np.std(action[i]))
                action_mean = np.array(action_mean)
                action_var = np.array(action_var)
                action_mean[1:-1] = (action_mean[:-2] + action_mean[1:-1] + action_mean[2:]) / 3
                action_var[1:-1] = (action_var[:-2] + action_var[1:-1] + action_var[2:]) / 3
                plt.fill_between(step, action_mean - action_var, action_mean + action_var,
                    alpha=0.25, facecolor=f, antialiased=True)
                plt.plot(step, action_mean,  c, label=l)
            plt.legend()
            plt.savefig(output_dir + filename[:-5] + "_wheel.png", dpi=200)
            plt.close()
        '''
        for text, logs, lim in zip (['distance', 'velocity'], [log_lat, log_vel], [[-2, 2], [0, 20]]):
            plt.figure()
            plt.xlabel('step')
            plt.ylabel(text)
            #plt.ylim(lim)
            plt.rc('font', size=12)

            log1_mean = []
            log1_var = []
            for i in range(10, 70):
                log1_mean.append(np.mean(logs[0][i]))
                log1_var.append(np.std(logs[0][i]))
            log1_mean = np.array(log1_mean)
            log1_var = np.array(log1_var)

            log2_mean = []
            log2_var = []
            for i in range(10, 70):
                log2_mean.append(np.mean(logs[1][i]))
                log2_var.append(np.std(logs[1][i]))
            log2_mean = np.array(log2_mean)
            log2_var = np.array(log2_var)

            plt.fill_between(np.arange(10, 70), log1_mean - log1_var,  log1_mean + log1_var,
                alpha=0.25, facecolor='blue', antialiased=True)
            plt.fill_between(np.arange(10, 70), log2_mean - log2_var,  log2_mean + log2_var,
                alpha=0.25, facecolor='red', antialiased=True)
            plt.plot(np.arange(10, 70), log1_mean,  'b--', label='SAC')
            plt.plot(np.arange(10, 70), log2_mean, 'r-', label='Ours')
            plt.legend()
            plt.savefig(output_dir + "task_" + str(task) + "_route_" + str(route) + "_" + text + ".png", dpi=300)
            plt.close()
        '''