# ----------
# Part Five
#
# This time, the sensor measurements from the runaway Traxbot will be VERY 
# noisy (about twice the target's stepsize). You will use this noisy stream
# of measurements to localize and catch the target.
#
# ----------
# YOUR JOB
#
# Complete the next_move function, similar to how you did last time. 
#
# ----------
# GRADING
# 
# Same as part 3 and 4. Again, try to catch the target in as few steps as possible.

from robot import *
from math import *
from matrix import *
import random

import matplotlib.pyplot as plt
ax = plt.subplot(111)

from copy import deepcopy


def measure_prob(x1, x2, noise_sigma):
    # compute errors
    error = 1.0
    for i in range(len(x1)):
        error_i = abs(x1[i] - x2[i])
        # update Gaussian
        error *= (exp(- (error_i ** 2) / (noise_sigma ** 2) / 2.0) /  
                    sqrt(2.0 * pi * (noise_sigma ** 2)))
    return error

def next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER = None):
    # This function will be called after each time the target moves. 

    # The OTHER variable is a place for you to store any historical information about
    # the progress of the hunt (or maybe some localization information). Your return format
    # must be as follows in order to be graded properly.

    # P_LENGTH = 200
    # noise_init_xy = 0.01
    # noise_init_dist = 0.5
    # noise_turn = 0.1
    # noise_dist = 0.2
    # prob_sigma = 1

    P_LENGTH = 1000
    noise_init_xy = 0.01
    noise_init_dist = 0.1
    noise_turn = 0.5
    noise_dist = 0.1
    prob_sigma = 2

    if not OTHER:
        OTHER = {}
        OTHER["p_list"] = []
        for i in range(P_LENGTH):
            r = robot(random.gauss(target_measurement[0], noise_init_xy),
                    random.gauss(target_measurement[1], noise_init_xy),
                    heading=(random.random()-0.5) * 2 * pi,
                    turning=(random.random()-0.5) * 2 * pi,
                    distance=(random.random()-0.5) * 2 * 3)
                    # distance=random.gauss(1.5, noise_init_dist))
            OTHER["p_list"].append(r)

    # prediction
    for p in OTHER["p_list"]:
        p.turning += random.gauss(0, noise_turn)
        p.turning = angle_trunc(p.turning)
        # # TODO
        p.distance += random.gauss(0, noise_dist)
        p.move(p.turning, p.distance)


    ax.scatter([x.x for x in OTHER["p_list"]], [x.y for x in OTHER["p_list"]], s=0.1)
    plt.pause(0.001)

    # measure
    w = []
    for p in OTHER["p_list"]:
        w.append(
            measure_prob((p.x, p.y), 
                         (target_measurement[0], target_measurement[1]), 
                         prob_sigma))

    # resampling
    p3 = []
    index = int(random.random() * P_LENGTH)
    beta = 0.0
    mw = max(w)

    for i in range(P_LENGTH):
        beta += random.random() * 2.0 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % P_LENGTH
        r = OTHER["p_list"][index]
        p = robot(x=r.x, y=r.y, heading=r.heading, distance=r.distance, turning=r.turning)
        p3.append(p)
    OTHER["p_list"] = p3

    # prediction
    # if OTHER.get("index", None) is None:
    #     OTHER["index"] = 0
    # OTHER["index"] += 1
    # print("index", OTHER["index"])

    p_list = deepcopy(OTHER["p_list"])
    for p in p_list:
        # if OTHER["index"] > 100:
        #     p.move(p.turning, p.distance)
        # else:
        #     p.move(p.turning, p.distance)
        #     p.move(p.turning, p.distance)
        p.move(p.turning, p.distance)

    # estimation
    xy_estimate = (sum([x.x for x in p_list]) / P_LENGTH,
                   sum([x.y for x in p_list]) / P_LENGTH)

    # hunter_position, hunter_heading, target_measurement
    distance = distance_between(hunter_position, xy_estimate) # full speed ahead!
    step = 1

    need_max = False
    while distance > max_distance:
        step += 1
        for p in p_list:
            p.move(p.turning, p.distance)

        # estimation
        xy_estimate = (sum([x.x for x in p_list]) / P_LENGTH,
                    sum([x.y for x in p_list]) / P_LENGTH)

        distance = distance_between(hunter_position, xy_estimate)
        distance -= (step-1) * max_distance
        
        need_max = True

    # print("-------step", step)
    if need_max:
        distance = max_distance

    heading_to_target = get_heading(hunter_position, xy_estimate)
    heading_difference = heading_to_target - hunter_heading
    turning =  heading_difference # turn towards the target
    turning = angle_trunc(turning)
    return turning, distance, OTHER

def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def demo_grading(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we 
    will grade your submission."""
    max_distance = 0.97 * target_bot.distance # 0.97 is an example. It will change.
    max_distance = 1000000.0
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0

    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:

        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print "You got it right! It took you ", ctr, " steps to catch the target."
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)
        
        # Don't try to move faster than allowed!
        # if distance > max_distance:
        #     distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()

        ctr += 1            
        if ctr >= 1000:
            print "It took too many steps to catch the target."
    return caught



def angle_trunc(a):
    """This maps all angles to a domain of [-pi, pi]"""
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi

def get_heading(hunter_position, target_position):
    """Returns the angle, in radians, between the target and hunter positions"""
    hunter_x, hunter_y = hunter_position
    target_x, target_y = target_position
    heading = atan2(target_y - hunter_y, target_x - hunter_x)
    heading = angle_trunc(heading)
    return heading

def naive_next_move(hunter_position, hunter_heading, target_measurement, max_distance, OTHER):
    """This strategy always tries to steer the hunter directly towards where the target last
    said it was and then moves forwards at full speed. This strategy also keeps track of all 
    the target measurements, hunter positions, and hunter headings over time, but it doesn't 
    do anything with that information."""
    if not OTHER: # first time calling this function, set up my OTHER variables.
        measurements = [target_measurement]
        hunter_positions = [hunter_position]
        hunter_headings = [hunter_heading]
        OTHER = (measurements, hunter_positions, hunter_headings) # now I can keep track of history
    else: # not the first time, update my history
        OTHER[0].append(target_measurement)
        OTHER[1].append(hunter_position)
        OTHER[2].append(hunter_heading)
        measurements, hunter_positions, hunter_headings = OTHER # now I can always refer to these variables
    
    heading_to_target = get_heading(hunter_position, target_measurement)
    heading_difference = heading_to_target - hunter_heading
    turning =  heading_difference # turn towards the target
    distance = max_distance # full speed ahead!
    return turning, distance, OTHER


# If you would like a visualization, you can can replace the demo_grading 
# function with the following code and run it locally. 
# Here, the red circle represents the measured position of the broken robot, 
# the blue arrow and line represent your chase robot, 
# and the green turtle and green line represent the true position of the broken robot.
def demo_grading_show(hunter_bot, target_bot, next_move_fcn, OTHER = None):
    """Returns True if your next_move_fcn successfully guides the hunter_bot
    to the target_bot. This function is here to help you understand how we 
    will grade your submission."""
    # max_distance = 0.97 * target_bot.distance # 0.98 is an example. It will change.
    max_distance = 1000000.0
    separation_tolerance = 0.02 * target_bot.distance # hunter must be within 0.02 step size to catch target
    caught = False
    ctr = 0
    #For Visualization
    import turtle
    window = turtle.Screen()
    window.bgcolor('white')
    chaser_robot = turtle.Turtle()
    chaser_robot.shape('arrow')
    chaser_robot.color('blue')
    chaser_robot.resizemode('user')
    chaser_robot.shapesize(0.3, 0.3, 0.3)
    broken_robot = turtle.Turtle()
    broken_robot.shape('turtle')
    broken_robot.color('green')
    broken_robot.resizemode('user')
    broken_robot.shapesize(0.3, 0.3, 0.3)
    size_multiplier = 15.0 #change size of animation
    chaser_robot.hideturtle()
    chaser_robot.penup()
    chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
    chaser_robot.showturtle()
    broken_robot.hideturtle()
    broken_robot.penup()
    broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
    broken_robot.showturtle()
    measuredbroken_robot = turtle.Turtle()
    measuredbroken_robot.shape('circle')
    measuredbroken_robot.color('red')
    measuredbroken_robot.penup()
    measuredbroken_robot.resizemode('user')
    measuredbroken_robot.shapesize(0.1, 0.1, 0.1)
    broken_robot.pendown()
    chaser_robot.pendown()
    #End of Visualization
    # We will use your next_move_fcn until we catch the target or time expires.
    while not caught and ctr < 1000:
        # Check to see if the hunter has caught the target.
        hunter_position = (hunter_bot.x, hunter_bot.y)
        target_position = (target_bot.x, target_bot.y)
        separation = distance_between(hunter_position, target_position)
        if separation < separation_tolerance:
            print "You got it right! It took you ", ctr, " steps to catch the target."
            caught = True

        # The target broadcasts its noisy measurement
        target_measurement = target_bot.sense()

        # This is where YOUR function will be called.
        turning, distance, OTHER = next_move_fcn(hunter_position, hunter_bot.heading, target_measurement, max_distance, OTHER)

        # Don't try to move faster than allowed!
        # if distance > max_distance:
        #     distance = max_distance

        # We move the hunter according to your instructions
        hunter_bot.move(turning, distance)

        # The target continues its (nearly) circular motion.
        target_bot.move_in_circle()
        #Visualize it
        measuredbroken_robot.setheading(target_bot.heading*180/pi)
        measuredbroken_robot.goto(target_measurement[0]*size_multiplier, target_measurement[1]*size_multiplier-100)
        measuredbroken_robot.stamp()
        broken_robot.setheading(target_bot.heading*180/pi)
        broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-100)
        chaser_robot.setheading(hunter_bot.heading*180/pi)
        chaser_robot.goto(hunter_bot.x*size_multiplier, hunter_bot.y*size_multiplier-100)
        #End of visualization
        ctr += 1            
        if ctr >= 1000:
            print "It took too many steps to catch the target."
    return caught

target = robot(0.0, 10.0, 0.0, 2*pi / 30, 1.5)
measurement_noise = 2.0*target.distance # VERY NOISY!!
# measurement_noise = 0.05*target.distance # VERY NOISY!!
target.set_noise(0.0, 0.0, measurement_noise)

hunter = robot(-10.0, -10.0, 0.0)

print demo_grading_show(hunter, target, next_move)





