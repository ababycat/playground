#!usr/bin/python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kalman import Kalman

fns = os.listdir()
for fn in fns:
    if fn.split('.')[-1] == 'csv':
        break
if fn.split('.')[-1] != 'csv':
    print('don\'t find csv file')
    quit()

df = pd.read_csv(fn)

df.set_index(pd.to_datetime(df.iloc[:, 0]) - pd.to_datetime(df.iloc[0, 0]), inplace=True)
df.drop(df.columns[0], axis=1, inplace=True)
df.set_index(df.index.total_seconds(), inplace=True)

data_length = np.shape(df)[0]

x_time = df.index

ax_data = df['ax'].as_matrix()
ay_data = df['ay'].as_matrix()
az_data = df['az'].as_matrix()

gx_data = df['gx'].as_matrix()
gy_data = df['gy'].as_matrix()
gz_data = df['gz'].as_matrix()

offset_num = 500
ax_offset = np.mean(ax_data[0:offset_num])
ay_offset = np.mean(ay_data[0:offset_num])
az_offset = np.mean(az_data[0:offset_num])
gx_offset = np.mean(gx_data[0:offset_num])
gy_offset = np.mean(gy_data[0:offset_num])
gz_offset = np.mean(gz_data[0:offset_num])

fi_kf = Kalman(F=np.array([[1, 1], [0, 1]]),
            P = np.array([[1000, 0], [0, 1000]]),
            Q = np.array([[0.01, 0], [0, 0.01]]),
            R = np.array([[100]]),
            H = np.array([[1, 0]]),
            x = np.array([[0], [0]]),
            u = np.array([[0], [0]]))
            
theta_kf = Kalman(F=np.array([[1, 1], [0, 1]]),
            P = np.array([[1000, 0], [0, 1000]]),
            Q = np.array([[0.1, 0], [0, 0.1]]),
            R = np.array([[10]]),
            H = np.array([[1, 0]]),
            x = np.array([[0], [0]]),
            u = np.array([[0], [0]]))
should_init_kf = True

fi_raw_list = []
theta_raw_list = []

fi_predict_list = []
theta_predict_list = []

# begin
for id_measure in range(offset_num, data_length):
    # get raw data
    ax_raw = ax_data[id_measure]
    ay_raw = ay_data[id_measure]
    az_raw = az_data[id_measure]
    gx_raw = gx_data[id_measure]
    gy_raw = gy_data[id_measure]
    gz_raw = gz_data[id_measure]
    
    # substract offset for gyro only
    gx_raw = gx_raw - gx_offset
    gy_raw = gy_raw - gy_offset
    gz_raw = gz_raw - gz_offset

    ax, ay, az, gx, gy, gz = ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw

    # get measure
    fi = np.arctan(ax/max(np.sqrt(np.sum(ay**2+az**2)), 1e-8))*180/np.pi
    theta = np.arctan(ay/max(az, 1e-8))*180/np.pi

    if should_init_kf:
        should_init_kf = False
        fi_kf.set_init_x = np.array([[fi], [0]])
        theta_kf.set_init_x = np.array([[theta], [0]])

    fi_kf.predict()
    fi_predict = fi_kf.measurement(np.array([[fi]]))[0, 0]

    theta_kf.predict()
    theta_predict = theta_kf.measurement(np.array([[theta]]))[0, 0]

    fi_raw_list.append(fi)
    theta_raw_list.append(theta)

    fi_predict_list.append(fi_predict)
    theta_predict_list.append(theta_predict)

plt.figure()
plt.plot(x_time[offset_num:], fi_raw_list)
plt.plot(x_time[offset_num:], fi_predict_list)
plt.title('fi')
plt.legend(['raw', 'predict'])

plt.figure()
plt.plot(x_time[offset_num:], theta_raw_list)
plt.plot(x_time[offset_num:], theta_predict_list)
plt.title('theta')
plt.legend(['raw', 'predict'])
plt.show()























