import os
import time
import datetime
import threading

import matplotlib.pyplot  as plt
from smbus2 import SMBusWrapper
import numpy as np
import pandas as pd

MPU6050_ADDRESS = 0X68
MPU6050_PWR_MGMT_1 = 0X6B
MPU6050_GYRO_CONFIG = 0X1B
MPU6050_ACCEL_CONFIG = 0X1C

g_lsb = 1./32.8
a_lsb = 1./4096

os.system("sudo chmod 666 /dev/i2c-1")

def printData(ax, ay, az, gx, gy, gz):
#    print('ax=%5.3f, ay=%5.3f, az=%5.3f, gx=%5.3f, gy=%5.3f, gz=%5.3f' %(ax, ay, az, gx, gy, gz))
    print('gx ={:>8.3f},    gy ={:>8.3f},   gz ={:>8.3f},   ax ={:>8.3f},   ay ={:>8.3f},   az ={:>8.3f}'.format(gx, gy, gz, ax, ay, az))

def readWord(addr, reg):
    h = bus.read_byte_data(MPU6050_ADDRESS, 0X43)
    l = bus.read_byte_data(MPU6050_ADDRESS, 0X44)
    return h<<8|l

def getRawData(data=[]):
    x = data[0] << 8 | data[1]
    y = data[2] << 8 | data[3]
    z = data[4] << 8 | data[5]
    x = x if x <= 0x8000 else -(0xffff - x + 1)
    y = y if y <= 0x8000 else -(0xffff - y + 1)
    z = z if z <= 0x8000 else -(0xffff - z + 1)
    return x, y, z
   
def getData(bus):
    gyro = getRawData(bus.read_i2c_block_data(MPU6050_ADDRESS, 0x43, 6))
    accel = getRawData(bus.read_i2c_block_data(MPU6050_ADDRESS, 0x3B, 6))
    gyro = [x*g_lsb for x in gyro]
    accel = [x*a_lsb for x in accel]
    return gyro, accel

def mpu6050Init(bus):
    bus.write_byte_data(MPU6050_ADDRESS, MPU6050_PWR_MGMT_1, 0x00)
    bus.write_byte_data(MPU6050_ADDRESS, MPU6050_GYRO_CONFIG, 0x10)#1000 o/S
    bus.write_byte_data(MPU6050_ADDRESS, MPU6050_ACCEL_CONFIG, 0x10)#8 g


class Tick(threading.Thread):
    def __init__(self, thread_id):
        threading.Thread.__init__(self)
        self._thread_id = thread_id
        self._time_up = False
        self._last = None        
        self._signal = threading.Event()
        self._release = False
        
    def run(self):
        while not self._release:
            self._time_up = True
            time.sleep(0.005)
            self._signal.set()
            self._signal.clear()        
    
    def wait(self):
        self._signal.wait()

    def release(self):
        self._release = True
    
def main():
    
    tick1 = Tick(1)
    tick1.start()

    ax_list = []
    ay_list = []
    az_list = []
    gx_list = []
    gy_list = []
    gz_list = []
    time_list = []

    with SMBusWrapper(1) as bus:
        mpu6050Init(bus)

        start = None
        for i in range(5000):
            tick1.wait()
            
            now = datetime.datetime.now()
            if start is None:
                start = now
            gyro, accel = getData(bus)

            ax, ay, az = accel
            gx, gy, gz = gyro
            
            ax_list.append(ax)
            ay_list.append(ay)
            az_list.append(az)
            gx_list.append(gx)
            gy_list.append(gy)
            gz_list.append(gz)
            time_list.append(now-start)
            if i > 1000:
                print('moving!')
            else:
                print('getting')
            
    tick1.release()
    tick1.join()

    df = pd.DataFrame({
        'ax': ax_list,
        'ay': ay_list,
        'az': az_list,
        'gx': gx_list,
        'gy': gy_list,
        'gz': gz_list}, index=time_list)
    df.to_csv('data'+ str(datetime.datetime.now()) +'.csv')
    print('save done')

if __name__ == '__main__':
    main()


