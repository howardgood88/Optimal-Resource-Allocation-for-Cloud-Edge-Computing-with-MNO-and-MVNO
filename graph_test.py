from utils import (beta, PT5)
from parameters import *
import numpy as np
import matplotlib.pyplot as plt
from task_deployment import UtilityFunc

test_list = [beta, PT5]

def beta_test():
    plt.figure()
    plt.title('Beta Distribution Testing')
    _slice = 10000
    digit = int(np.log10(_slice / beta_t))
    x = np.arange(beta_d, beta_t + beta_d, beta_t / _slice)
    y = np.zeros((_slice))
    for _ in range(_slice):
        y[int(round(beta(beta_a, beta_b, beta_t, beta_d) / 1000 - beta_d, digit) * 10 ** digit)] += 1
    plt.plot(x, y)

def PT5_test():
    plt.figure()
    plt.title('PT5 Distribution Testing')
    _slice = 2000
    max_x = 20
    digit = int(np.log10(_slice / max_x))
    x = np.arange(PT5_cloud_d, max_x + PT5_cloud_d + (max_x - 1) / _slice, max_x / _slice)
    y = np.zeros((_slice + 1))
    for _ in range(_slice):
        idx = int((round(PT5(PT5_cloud_a, PT5_cloud_b, PT5_cloud_d), digit) - PT5_cloud_d) * 10 ** digit)
        val = idx / 10 ** digit
        y[idx] += 1
    plt.plot(x, y)

def utility_func_bw_test(max_x, _slice, func):
    plt.figure()
    plt.title(f'{func.__qualname__} Utility Function Testing')
    x = np.arange(max_x / _slice, max_x, max_x / _slice)
    y = []
    for val in x:
        y.append(func(val))
    plt.plot(x, y)

def utility_func_delay_test(max_x, _slice, func, location):
    plt.figure()
    plt.title(f'{func.__qualname__} {location} Utility Function Testing')
    if location == 'cloud':
        d = PT5_cloud_d
    else:
        d = PT5_edge_d
    x = np.arange(d, max_x + d, max_x / _slice)
    y = []
    for val in x:
        y.append(func(val, location))
    plt.plot(x, y)
    
beta_test()
PT5_test()
plt.show()
utility_func_bw_test(800, 8000, UtilityFunc.VoIP.bw_up)
utility_func_bw_test(40, 4000, UtilityFunc.VoIP.bw_down)
utility_func_delay_test(20, 2000, UtilityFunc.VoIP.delay, 'cloud')
utility_func_delay_test(20, 2000, UtilityFunc.VoIP.delay, 'edge')
plt.show()
utility_func_bw_test(50, 5000, UtilityFunc.IPVideo.bw_up)
utility_func_bw_test(1000, 10000, UtilityFunc.IPVideo.bw_down)
plt.show()
utility_func_bw_test(50, 5000, UtilityFunc.FTP.bw_up)
utility_func_bw_test(1500, 15000, UtilityFunc.FTP.bw_down)
plt.show()