import random
from key_constants import Keys

from constants import action_list, key_list
from util.key_windows_function import press_key1, release_key
import numpy as np
import time

'''
function not used after the first time, i needed it to get the first time the constant list key, im commenting it
to keep it here
'''
import win32api


# def get_key():
#     keys = []
#     for key in key_list:
#         if win32api.GetAsyncKeyState(ord(key)):
#             keys.append(key)
#     return keys

def from_vector_to_keys(vec):  # vector converted in key argument, easy to use

    assert len(vec) == 6
    index = np.argmax(vec)
    return action_list[index]


def press_key(key, duration=None):  # pressing and releasing the key, sleep to control the duration of pressing.

    press_key1(key)
    if duration is not None:
        time.sleep(duration)
        release_key(key)


def control(key,
            duration):  # all type of control, Deep GTA V was better in this case, duration default none, check function above

    if 'w' == key:
        release_key(Keys.s)
        release_key(Keys.a)
        release_key(Keys.d)
        press_key(Keys.w)
    elif 'wa' == key:
        release_key(Keys.s)
        release_key(Keys.d)
        press_key(Keys.w)
        press_key(Keys.a, duration=duration)
    elif 'wd' == key:
        release_key(Keys.s)
        release_key(Keys.a)
        press_key(Keys.w)
        press_key(Keys.d, duration=duration)
    elif 'sa' == key:
        release_key(Keys.w)
        release_key(Keys.d)
        press_key(Keys.s)
        press_key(Keys.a, duration=duration)
    elif 'sd' == key:
        release_key(Keys.w)
        release_key(Keys.a)
        press_key(Keys.s)
        press_key(Keys.d, duration=duration)
    elif 's' == key:
        release_key(Keys.w)
        release_key(Keys.a)
        release_key(Keys.d)
        press_key(Keys.s)
    else:
        print("get wrong action keys : {}".format(key))


def vehicle_control_movement(vec, duration):  # simply the actual function who use a vector to control the car ingame

    keys = from_vector_to_keys(vec)
    control(keys, duration)


def random_action():  # random car action, the vector is similar to this  [[0,0,0,1,0,0]]

    action_index = random.randint(0, 5)
    temp_actions = np.zeros((1, 6))
    temp_actions[0][action_index] = 1
    return temp_actions


def back_to_start():
    # built-in function that will send the car back to the initial point where it spawned when the training started

    press_key(Keys.h)
    print("back")
    release_key(Keys.h)
