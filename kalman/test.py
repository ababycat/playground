#!urs/bin/python3
#hello encoding=utf-8

import time
import datetime

import threading
import os
import sys


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

    last = None
    i = 0    
    while i < 1000:
        tick1.wait()
        i = i + 1
        print('done')
        
        now = datetime.datetime.now()
        if last != None:
            print(now - last)
        last = now
    
    tick1.release()
    tick1.join()

if __name__ == '__main__':
    main()

#class mainThread(threading.Thread):
#    def __init__(self, thread_id):
#        threading.Thread.__init__(self)
#        self._thread_id = thread_id

#    def run(self):
#        while True:
#            if 


#tick1 = Tick(1)
#tick1.start()

#output_list = []

#i = 0
#last_time = None
#while i < 1000:
#    if tick1.time_up():
#        i = i + 1
#        this_time = datetime.datetime.now()
#        if last_time is not None:
#            print(this_time - last_time)
#        last_time = this_time 



#class _Timer(threading.Thread):
#    def __init__(self, interval, function, args=[], kwargs={}):
#        threading.Thread.__init__(self)
#        self.interval = interval 
#        self.function = function
#        self.args = args
#        self.kwargs = kwargs
#        self.finished = threading.Event()

#    def cancel(self):
#        self.finished.set()

#    def run(self):
#        self.finished.wait(self.interval) 
#        if not self.finished.is_set():
#            self.function(*self.args, **self.kwargs)
##        self.finished.set()
#        
#class LoopTimer(_Timer):
#    def __init__(self, interval, function, args=[], kwargs={}):
#        _Timer.__init__(self, interval, function, args, kwargs)

#    def run(self):
#        while True:
#            if not self.finished.is_set():
#                self.finished.wait(self.interval)
#                self.function(*self.args, **self.kwargs) 
#            else:
#                break

#start = None
#def testlooptimer():
#    global start
#    if start is not None:
#        now = datetime.datetime.now()
#        print(now - start)
#        start = now
#    else:
#        start = datetime.datetime.now()
#        print(start)

#t = _Timer(0.1, testlooptimer)
#t.start()
#time.sleep(3)
#t.cancel()



#if __name__ == '__main__':
#    t = LoopTimer(0.005, testlooptimer)
#    t.start()
#    time.sleep(5)
#    t.cancel()


#import time  
#from threading import Timer  
#  
#def print_time( enter_time ):  
#    print "now is", time.clock() , "enter_the_box_time is", enter_time  
#  
#  
#print time.time()  
#Timer(0.2,  print_time, ( time.clock(), )).start()  
#Timer(0.2, print_time, ( time.clock(), )).start()  
#print time.time()  

#import threading
#import time 
#    
#class MyThread(threading.Thread): 
#    def __init__(self, signal): 
#        threading.Thread.__init__(self)
#        # 初始化
#        self.singal = signal 
#            
#    def run(self): 
#        print(self.name)
#        # 进入等待状态
#        self.singal.wait() 
#        print(self.name)
#            
#if __name__ == "__main__":
#    # 初始 为 False
#    singal = threading.Event() 
#    for t in range(0, 3): 
#        thread = MyThread(singal) 
#        thread.start()
#        
#    print("main thread sleep 3 seconds... " )
#    time.sleep(3) 
#    #　唤醒含有signal,　处于等待状态的线程 
#    singal.set()
