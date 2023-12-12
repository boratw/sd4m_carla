import numpy as np
import json
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import itertools


log_dir = 'test_log/Train3_2/break_test/'
#filenames = os.listdir(log_dir)
#for filename in filenames:
#    if filename[-4:] == 'json':
task = 0
for route in [0]:
    log_lat = []
    log_zt = []
    for control in [1, 0]:
        filename = "task_" + str(task) + "_control_" + str(control) + "_route_" + str(route) + ".json"
        with open(log_dir + filename) as f:
            json_object = json.load(f)
        
        traj_map = np.full((4096, 8192, 3), 0, np.uint8)
            #cv2.polylines(traj_map, np.array([np.array(route) * 128. + np.array((-2048, 2048))], np.int32), False, (0, 0, 0))
        for i, r in enumerate(json_object['pos']):
            overlay = np.full((4096, 8192, 3), 0, np.uint8)
            cv2.polylines(overlay, np.array([np.array(r) * 64. + np.array((-64 * 64., 0.))+ np.array((4096, 2048))], np.int32), False, (255, 255, 255), 8)
            traj_map = cv2.addWeighted(overlay, 0.05, traj_map, 1., 0)
        cv2.imwrite(log_dir + filename[:-5] + ".png", traj_map)



        lat = []
        zt = []
        distance = json_object['distance']
        ztask = json_object['zt']
        lastindex = []
        start_index = 0
        end_index = 0
        for i, r in enumerate(json_object['routeit']):
            readed_it = 0
            for j, t in enumerate(r):
                if readed_it != t:
                    while t >= len(lat):
                        lat.append([])
                        zt.append([])
                    lat[t].append(distance[i][j])
                    zt[t].append(ztask[i][j])
                    readed_it = t
                if t == 20:
                    start_index = j
                elif t == 50:
                    end_index = j
            lastindex.append(readed_it)
            

        log_lat.append(lat[1:])
        log_zt.append(zt[1:])
        
        print(start_index, end_index)

        plt.figure()
        end_index = (end_index - start_index)  // 20 * 20 + start_index
        step = np.arange(start_index, end_index, 20 )
        plt.xlabel('step')
        plt.ylabel('wheel')
        plt.ylim([-0.5, 1.0])
        plt.rc('font', size=12)
        log_action = np.array(json_object['action'])[:, start_index:end_index]
        log_action = np.stack([log_action[:, :, 0] * 0.6 + log_action[:, :, 2] * 0.4,
                        log_action[:, :, 1] * 0.6 + log_action[:, :, 3] * 0.4,
                        log_action[:, :, 2] * 0.6 + log_action[:, :, 0] * 0.4,
                        log_action[:, :, 3] * 0.6 + log_action[:, :, 1] * 0.4], axis=2)
        log_action_mean = np.mean(log_action, axis=0)
        log_action_var = np.std(log_action, axis=0)
        log_action_mean = log_action_mean.reshape((-1, 20, 4))
        log_action_var = log_action_var.reshape((-1, 20, 4))
        log_action_mean = np.mean(log_action_mean, axis=1)
        log_action_var = np.mean(log_action_var, axis=1)
        plt.fill_between(step, log_action_mean[:, 0] - log_action_var[:, 0],  log_action_mean[:, 0] + log_action_var[:, 0],
            alpha=0.25, facecolor='red', antialiased=True)
        plt.fill_between(step, log_action_mean[:, 1] - log_action_var[:, 1],  log_action_mean[:, 1] + log_action_var[:, 0],
            alpha=0.25, facecolor='green', antialiased=True)
        plt.fill_between(step, log_action_mean[:, 2] - log_action_var[:, 2],  log_action_mean[:, 2] + log_action_var[:, 0],
            alpha=0.25, facecolor='blue', antialiased=True)
        plt.fill_between(step, log_action_mean[:, 3] - log_action_var[:, 3],  log_action_mean[:, 3] + log_action_var[:, 0],
            alpha=0.25, facecolor='cyan', antialiased=True)
        plt.plot(step, log_action_mean[:, 0],  'ro-', label='FL')
        plt.plot(step, log_action_mean[:, 1], 'g^-', label='FR')
        plt.plot(step, log_action_mean[:, 2], 'b*-', label='RL')
        plt.plot(step, log_action_mean[:, 3], 'cs-', label='RR')
        plt.legend()
        plt.savefig(log_dir + filename[:-5] +"_wheel.png", dpi=200)
        
    lastindex = []
    log_lat_mean = []
    log_lat_var = []
    log_zt_mean = []
    log_zt_var = []
    for i in range(2):
        log_lat[i] = [x for x in log_lat[i][20:] if len(x) == 25]
        log_zt[i] = [x for x in log_zt[i][20:] if len(x) == 25]
        if len(log_lat[i]) > 50:
            log_lat[i] = log_lat[i][:50]
            log_zt[i] = log_zt[i][:50]
        lastindex.append(len(log_lat[i]))
        log_lat[i] = np.array(log_lat[i])
        log_lat_mean.append(np.mean(log_lat[i], axis=1))
        log_lat_var.append(np.std(log_lat[i], axis=1))
        log_zt_mean.append(np.mean(log_zt[i], axis=1))
        log_zt_var.append(np.std(log_zt[i], axis=1))


    plt.figure()
    step = [np.arange(x) for x in lastindex]
    plt.xlabel('step')
    plt.ylabel('lateral distance')
    plt.fill_between(step[0], log_lat_mean[0] - log_lat_var[0],  log_lat_mean[0] + log_lat_var[0],
        alpha=0.25, facecolor='blue', antialiased=True)
    plt.fill_between(step[1], log_lat_mean[1] - log_lat_var[1],  log_lat_mean[1] + log_lat_var[1],
        alpha=0.25, facecolor='red', antialiased=True)
    plt.plot(step[0], log_lat_mean[0],  'b--', label='SAC')
    plt.plot(step[1], log_lat_mean[1], 'r-', label='Ours')
    plt.legend()
    plt.savefig(log_dir + filename[:-5] +"_dist.png", dpi=300)

    plt.figure()
    step = [np.arange(x) for x in lastindex]
    plt.xlabel('step')
    plt.ylabel('za')
    plt.fill_between(step[1], log_zt_mean[1] - log_zt_var[1] / 2,  log_zt_mean[1] + log_zt_var[1] / 2,
        alpha=0.25, facecolor='blue', antialiased=True)
    plt.plot(step[1], log_zt_mean[1],  'b--', label='za')
    plt.legend()
    plt.savefig(log_dir + filename[:-5] +"_za.png", dpi=300)
        
'''
        step = np.arange(0, lastindices[0] )
        plt.figure()
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Lateral Error')
        ax1.set_ylim([-2., 2.])
        log_lat = np.array(json_object['distance'][0])[:lastindices[0]]
        line1 = ax1.plot(step, log_lat, 'r-', label='Lateral Error')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Task Latent')
        ax2.set_ylim([-2., 2.])
        log_zt = np.array(json_object['task'][0])[:lastindices[0]]
        log_zt[log_zt<-2.56] = -2.56
        line2 = ax2.plot(step, log_zt, 'b-', label='Task Latent')

        lines = line1 + line2
        plt.legend(lines, [l.get_label() for l in lines])
        plt.savefig(log_dir + filename[:-5] +"_zt.png", dpi=200)

        
        
        step = np.arange(0, lastindex)
        plt.figure()
        plt.xlabel('step')
        plt.ylabel('lateral distance')
        plt.ylim([-3., 3.])
        for i, r in enumerate(json_object['pos']):
            r_array = np.array(r[:lastindices[i]])[:, 1] + np.arange(0, lastindex) * 0.001 + 204.3
            #r_array[:150] *= 0.5
            plt.plot(step, r_array, 'r-')
        plt.savefig('test_log/Train2_1/231011_traj_test/lat_'+ str(task) + '_control_' + str(control) + '_route_' + str(routeit) + ".png", dpi=200)

        plt.figure()
        plt.xlabel('step')
        plt.ylabel('va')
        plt.ylim([-3., 3.])
        for i, r in enumerate(json_object['pos']):
            r_array = []
            for j in r[:lastindices[i]]:

            np.array(r[:lastindices[i]])[:, 1] + np.arange(0, lastindex) * 0.001 + 204.3
            r_array[:150] *= 0.5
            plt.plot(step, r_array, 'r-')
        plt.savefig('test_log/Train2_1/231011_traj_test/lat_'+ str(task) + '_control_' + str(control) + '_route_' + str(routeit) + ".png", dpi=200)
'''



