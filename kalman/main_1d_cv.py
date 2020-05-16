#!usr/bin/python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2


class Kalman:
    def __init__(self,
                 F=np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                             [0, 0, 1, 0], [0, 0, 0, 1]]),
                 P=np.array([[1000, 0, 0, 0], [0, 1000, 0, 0],
                             [0, 0, 1000, 0], [0, 0, 0, 1000]]),
                 Q=np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0],
                             [0, 0, 0, 0], [0, 0, 0, 0]]),
                 R=np.array([[10, 0], [0, 10]]),
                 H=np.array([[1, 0, 0, 0], [0, 1, 0, 0]]),
                 x=np.array([[0], [0], [0], [0]]),
                 u=np.array([[0], [0], [0], [0]]), use_opencvKF=False):
        self._F = F
        self._P = P
        self._u = u
        self._Q = Q
        self._R = R
        self._H = H
        self._x = x

        self._x_pred = None
        self._z_pred = None
        self._z = None

        self.use_opencvKF = use_opencvKF
        if self.use_opencvKF:
            self.kf = cv2.KalmanFilter(F.shape[0], H.shape[0])
            self.kf.transitionMatrix = self._F.copy().astype(np.float32)
            self.kf.measurementMatrix = self._H.copy().astype(np.float32)
            self.kf.measurementNoiseCov = self._R.copy().astype(np.float32)
            self.kf.processNoiseCov = self._Q.copy().astype(np.float32)
            self.kf.statePost = self._x.copy().astype(np.float32)
            self.kf.errorCovPost = self._P.copy().astype(np.float32)

    def predict(self):
        if self.use_opencvKF:
            self._x_pred = self.kf.predict()
            self._z_pred = self.kf.measurementMatrix.dot(self._x_pred)
            return self._z_pred
        else:
            self._x_pred = self._F.dot(self._x)  # + self._u
            self._P = self._F.dot(self._P).dot(self._F.transpose()) + self._Q
            self._z_pred = self._H.dot(self._x_pred)
            return self._z_pred

    def measurement(self, measure):
        if self.use_opencvKF:
            self._x = self.kf.correct(measure.astype(np.float32))
            self._z = self.kf.measurementMatrix@self._x
        else:
            z = measure.copy()
            K = self._P.dot(self._H.transpose()).dot(np.linalg.inv(
                self._H.dot(self._P).dot(self._H.transpose()) + self._R))
            # self._x = self._x_pred + K.dot(z - self._H.dot(self._x_pred))
            self._x = self._x_pred + K.dot(z - self._z_pred)
            self._P = (np.eye(K.shape[0])-K.dot(self._H)).dot(self._P)
            self._z = self._H.dot(self._x)

    def get_status(self):
        return self._x

    def get_status_o(self):
        return self._z


fns = os.listdir()
for fn in fns:
    if fn.split('.')[-1] == 'csv':
        break
if fn.split('.')[-1] != 'csv':
    print('don\'t find csv file')
    quit()

df = pd.read_csv(fn)

df.set_index(pd.to_datetime(df.iloc[:, 0]) -
             pd.to_datetime(df.iloc[0, 0]), inplace=True)
df.drop(df.columns[0], axis=1, inplace=True)
df.set_index(df.index.total_seconds(), inplace=True)

data_length = np.shape(df)[0]

x_time = df.index

ax_data = df['ax'].values
ay_data = df['ay'].values
az_data = df['az'].values

gx_data = df['gx'].values
gy_data = df['gy'].values
gz_data = df['gz'].values

offset_num = 500
ax_offset = np.mean(ax_data[0:offset_num])
ay_offset = np.mean(ay_data[0:offset_num])
az_offset = np.mean(az_data[0:offset_num])
gx_offset = np.mean(gx_data[0:offset_num])
gy_offset = np.mean(gy_data[0:offset_num])
gz_offset = np.mean(gz_data[0:offset_num])

# q = 1000
# r = 0.001

Q_list = [1]
R_list = [100]
for q, r in zip(Q_list, R_list):
    F = np.array([[1, 1], [0, 1]])
    P = np.array([[10, 0], [0, 10]])
    Q = np.array([[q, 0], [0, 0]])
    R = np.array([[r]])
    H = np.array([[1, 0]])
    x = np.array([[0], [0]])

    F, H, P, Q, R = [x.astype(np.float32) for x in [F, H, P, Q, R]]
    
    fi_kf = cv2.KalmanFilter(2, 1)
    fi_kf.transitionMatrix = F.copy()
    fi_kf.measurementMatrix = H.copy()
    fi_kf.measurementNoiseCov = R.copy()
    fi_kf.processNoiseCov = Q.copy()
    # kf.statePost = x_init.copy()
    fi_kf.errorCovPost = P.copy()

    theta_kf = cv2.KalmanFilter(2, 1)
    theta_kf.transitionMatrix = F.copy()
    theta_kf.measurementMatrix = H.copy()
    theta_kf.measurementNoiseCov = R.copy()
    theta_kf.processNoiseCov = Q.copy()
    # kf.statePost = x_init.copy()
    theta_kf.errorCovPost = P.copy()

    should_init_kf = False

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
            fi_kf.statePost = np.array([[fi], [0]]).astype(np.float32)
            theta_kf.statePost = np.array([[theta], [0]]).astype(np.float32)

        fi_kf.predict()
        fi_predict = fi_kf.correct(np.array([[fi]]).astype(np.float32))[0, 0]

        theta_kf.predict()
        theta_predict = theta_kf.correct(np.array([[theta]]).astype(np.float32))[0, 0]

        fi_raw_list.append(fi)
        theta_raw_list.append(theta)

        fi_predict_list.append(fi_predict)
        theta_predict_list.append(theta_predict)

    plt.figure()
    plt.plot(x_time[offset_num:], fi_raw_list)
    plt.plot(x_time[offset_num:], fi_predict_list)
    plt.title('fi'+'q_'+str(q)+' r_'+str(r))
    plt.legend(['raw', 'predict'])

    plt.figure()
    plt.plot(x_time[offset_num:], theta_raw_list)
    plt.plot(x_time[offset_num:], theta_predict_list)
    plt.title('theta'+'q_'+str(q)+' r_'+str(r))
    plt.legend(['raw', 'predict'])
    # plt.show()


plt.show()
