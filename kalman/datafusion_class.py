#!usr/bin/python3

import math
import numpy as np
import matplotlib.pyplot as plt

from kalman import EKF
import particlelist as ps

# frequency = 250Hz
delt_t = 0.004
mass = 1.
kx_divide_m = 0.001
ky_divide_m = 0.001
r_delta = 1
theta_delta = 0.0
ax_delta = 0.0
ay_delta = 0.0
g = 9.8

init_x0, init_y0 = -10.0, 10.0
init_vx0, init_vy0 = 10., 0.0

samples_num = 500

kx = kx_divide_m
ky = ky_divide_m

np.random.seed(18)

def get_theta_r(x, y, x0=0., y0=0., noise_r=0., noise_theta=0.):
    r = math.sqrt((x-x0)**2+(y-y0)**2) + noise_r*np.random.randn()
    theta = math.atan2(y-y0, x-x0) + noise_theta*np.random.randn()
    return r, theta


class Throw(ps.Partical):
    def __init__(self, init_x=0.0, init_y=0.0, x_sp=.0, y_sp=.0, mass=.1, delt_t=delt_t):
        super().__init__(init_x=init_x, init_y=init_y, x_sp=x_sp, y_sp=y_sp, mass=mass, delt_t=delt_t)
        self._force = np.array([0,-mass*g])
        
    def apply_force(self):
        mg = -mass*g
        fx = -kx*self._speed[0]**2*np.sign(self._speed[0])
        fy = -ky*self._speed[1]**2*np.sign(self._speed[1])
        force = np.array([fx, mg+fy])
        self._force += force
    
    def get_data(self):
        return get_theta_r(self._location[0], self._location[1])
    
class Throw_noise(ps.Partical):
    def __init__(self, init_x=.0, init_y=.0, x_sp=.0, y_sp=.0, mass=.1, delt_t=delt_t):
        super().__init__(init_x=init_x, init_y=init_y, x_sp=x_sp, y_sp=y_sp, mass=mass, delt_t=delt_t)
        self._force = np.array([0,-mass*g])
        
    def apply_force(self):
        mg = -mass*g
        fx = -kx*self._speed[0]**2*np.sign(self._speed[0])
        fy = -ky*self._speed[1]**2*np.sign(self._speed[1])
        force = np.array([fx, mg+fy])
        self._force += force
    
    def get_data(self):
        return get_theta_r(self._location[0], self._location[1], noise_r=r_delta, noise_theta=theta_delta)


def generate_measurements():
    # initial position
    x0, y0 = init_x0, init_y0
    vx0, vy0 = init_vx0, init_vy0

    throw = Throw(init_x=x0, init_y=y0, x_sp=vx0, y_sp=vy0, mass=mass, delt_t=delt_t)
    throw_noise = Throw_noise(init_x=x0, init_y=y0, x_sp=vx0, y_sp=vy0, mass=mass, delt_t=delt_t)
    
    pos_real = np.zeros((samples_num, 2))
    pos_noise = np.zeros((samples_num, 2))

    pos_real[0, :] = get_theta_r(throw.location()[0], throw.location()[1])
    pos_noise[0, :] = get_theta_r(throw_noise.location()[0], throw.location()[1])

#    loc = []
    for idx in range(1, samples_num):
        throw.update()
        throw.apply_force()
        pos_real[idx, :] = throw.get_data()        
#        loc.append([throw._location[0], throw._location[1]])
        
        throw_noise.update()
        throw_noise.apply_force()
        pos_noise[idx, :] = throw_noise.get_data()        
#    loc = np.array(loc)
#    plt.plot(loc[:, 0], loc[:, 1])
#    plt.draw()
    
    return pos_real, pos_noise
    
    

class throw_EKF(EKF):
    def __init__(self):
        super().__init__(self, 
#            P = np.array([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100]]),
#            Q = np.array([[0, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.1]]),
#            R = np.array([[0.001, 0], [0, 0.001]]),
#            x = np.array([[0.1], [0.1], [0.1], [0.1]]))
            P = np.array([[1000., 0, 0, 0], [0, 1000, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1000]]),
            Q = np.array([[0.0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 3]]),
            R = np.array([[1, 0], [0, 0.01]]),
            x = np.array([[0.], [0], [0], [0]]))
            
    def predict(self):
        self._F = np.array([[0,                     delt_t, 0, 0], 
                            [0, -2*kx*self._x[1, 0]*delt_t, 0, 0], 
                            [0,                          0, 0, delt_t], 
                            [0, 0, 0, 2*ky*self._x[3, 0]*delt_t]]) + np.eye(4)

        self._H = np.array([[self._x[0, 0]/np.sqrt(self._x[0, 0]**2+self._x[2, 0]**2), 0,  self._x[2, 0]/np.sqrt(self._x[0, 0]**2+self._x[2, 0]**2), 0], 
                            [self._x[2, 0]/(self._x[0, 0]**2+self._x[2, 0]**2),        0,        -self._x[0, 0]/(self._x[0, 0]**2+self._x[2, 0]**2), 0]])

        self._x_pred[0, 0] = self._x[0, 0] + self._x[1, 0]*delt_t
        self._x_pred[1, 0] = self._x[1, 0] + (-kx*self._x[1, 0]**2)*delt_t
        self._x_pred[2, 0] = self._x[2, 0] + self._x[3, 0]*delt_t
        self._x_pred[3, 0] = self._x[3, 0] + (ky*self._x[3, 0]**2-g)*delt_t

        self._P = self._F.dot(self._P).dot(self._F.transpose()) + self._Q
        print(self._P)
        quit()
        return self._x_pred

    def measurement(self, measure):

        z = measure
        K = self._P.dot(self._H.transpose()).dot(np.linalg.inv(self._H.dot(self._P).dot(self._H.transpose()) + self._R))
        h_u = np.zeros((2, 1))
        h_u[:, 0] = get_theta_r(self._x_pred[0, 0], self._x_pred[2, 0])
        self._x = self._x_pred + K.dot(z - h_u)

        self._P = (np.eye(K.shape[0])-K.dot(self._H)).dot(self._P)
        return self._x
        
def main():
    pos_real, pos_noise = generate_measurements()

    kf = throw_EKF()
    pos_filter = np.zeros((samples_num, 2))
    pos_h_u = np.zeros((samples_num, 2))
    
    for idx in range(samples_num):
        measure = pos_real[idx, :].reshape(2,1)
        if idx==0:
            kf.set_init_x(np.array([[-10.], [0.], [10.], [0.]]))
#            continue
        kf.predict()
        estimate = kf.measurement(measure)
        
        pos_filter[idx, 0], pos_filter[idx, 1] = estimate[0, 0], estimate[2, 0]

    plt.figure()
    plt.subplot(projection='polar')
    plt.plot(pos_real[:, 1], pos_real[:, 0], 'o', markersize=1)
#    plt.plot(pos_noise[:, 1], pos_noise[:, 0], 'o', markersize=1)
#    plt.legend(['real', 'noise'])

    plt.figure()
    plt.plot(pos_filter[:, 0], pos_filter[:, 1], 'o', markersize=1)

    plt.show()

if __name__ == "__main__":
    main()





























