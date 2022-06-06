from utils import (beta, PT5)
from parameters import *
import numpy as np
import matplotlib.pyplot as plt
from task_deployment import UtilityFunc
from data.parameters import *
import os

dir = './figs/distributions/'
if not os.path.exists(dir):
    os.makedirs(dir)
figsize = (16, 12)

def plot_beta(a, b, t, d, _slice):
    digit = int(np.log10(_slice / t))
    x = np.arange(d, t + d, t / _slice)
    y = np.zeros((_slice))
    for _ in range(_slice):
        y[int(round(beta(a, b, t, d) / 1000 - d, digit) * 10 ** digit)] += 1
    plt.plot(x, y)

def beta_test():
    # user to vm bw
    plt.figure(figsize=figsize)
    plt.title('User to VM bandwidth generation (beta distribution)')
    plt.xlabel('bandwidth(Mbps)')
    plt.ylabel('times')
    plot_beta(beta_a, beta_b, beta_t, beta_d, 10000)
    plt.savefig(f'{dir}user_to_vm_bw')
    # voip uplink throughput
    plt.figure(figsize=figsize)
    plt.title('Task uplink bandwidth generation (beta distribution)')
    plt.xlabel('bandwidth(Kbps)')
    plt.ylabel('times')
    plot_beta(*voip_bw_up_attr, 10000)
    plt.savefig(f'{dir}voip_bw_up')
    # voip downlink throughput
    plt.figure(figsize=figsize)
    plt.title('Task downlink bandwidth generation (beta distribution)')
    plt.xlabel('bandwidth(Kbps)')
    plt.ylabel('times')
    plot_beta(*voip_bw_down_attr, 10000)
    plt.savefig(f'{dir}voip_bw_down')
    # IP VIdeo uplink throughput
    plt.figure(figsize=figsize)
    plt.title('IP VIdeo uplink bandwidth generation (beta distribution)')
    plt.xlabel('bandwidth(Kbps)')
    plt.ylabel('times')
    plot_beta(*ipVideo_bw_up_attr, 10000)
    plt.savefig(f'{dir}ipVideo_bw_up')
    # IP VIdeo downlink throughput
    plt.figure(figsize=figsize)
    plt.title('IP VIdeo downlink bandwidth generation (beta distribution)')
    plt.xlabel('bandwidth(Kbps)')
    plt.ylabel('times')
    plot_beta(*ipVideo_bw_down_attr, 10000)
    plt.savefig(f'{dir}ipVideo_bw_down')
    # FTP uplink throughput
    plt.figure(figsize=figsize)
    plt.title('FTP uplink bandwidth generation (beta distribution)')
    plt.xlabel('bandwidth(Kbps)')
    plt.ylabel('times')
    plot_beta(*ftp_bw_up_attr, 10000)
    plt.savefig(f'{dir}ftp_bw_up')
    # FTP downlink throughput
    plt.figure(figsize=figsize)
    plt.title('FTP downlink bandwidth generation (beta distribution)')
    plt.xlabel('bandwidth(Kbps)')
    plt.ylabel('times')
    plot_beta(*ftp_bw_down_attr, 10000)
    plt.savefig(f'{dir}ftp_bw_down')

def plot_PT5(a, b, d, _slice):
    max_x = 20
    digit = int(np.log10(_slice / max_x))
    x = np.arange(d, max_x + d + (max_x - 1) / _slice, max_x / _slice)
    y = np.zeros((_slice + 1))
    for _ in range(_slice):
        idx = int((round(PT5(a, b, d), digit) - d) * 10 ** digit)
        y[idx] += 1
    plt.plot(x, y)

def PT5_test():
    # cloud
    plt.figure(figsize=figsize)
    plt.xlabel('delay(ms)')
    plt.ylabel('times')
    plt.title('User to cloud VM bandwidth generation (PT5 distribution)')
    plot_PT5(PT5_cloud_a, PT5_cloud_b, PT5_cloud_d, 20000)
    plt.savefig(f'{dir}user_to_vm_cloud_delay')
    # edge
    plt.figure(figsize=figsize)
    plt.xlabel('delay(ms)')
    plt.ylabel('times')
    plt.title('User to edge VM bandwidth generation (PT5 distribution)')
    plot_PT5(PT5_edge_a, PT5_edge_b, PT5_edge_d, 20000)
    plt.savefig(f'{dir}user_to_vm_edge_delay')

def utility_func_bw_test(max_x, _slice, func):
    plt.figure(figsize=figsize)
    plt.title(f'{func.__qualname__} Utility Function')
    plt.xlabel('bandwidth(Kbps)')
    plt.ylabel('utility')
    x = np.arange(max_x / _slice, max_x, max_x / _slice)
    y = []
    for val in x:
        y.append(func(val))
    plt.plot(x, y)
    plt.savefig(f'{dir}{func.__qualname__}_utility_func.png')

def utility_func_delay_test(max_x, _slice, func, location):
    plt.figure(figsize=figsize)
    plt.title(f'{func.__qualname__} {location} Utility Function')
    plt.xlabel('delay(ms)')
    plt.ylabel('times')
    if location == 'cloud':
        d = PT5_cloud_d
    else:
        d = PT5_edge_d
    x = np.arange(d, max_x + d, max_x / _slice)
    y = []
    for val in x:
        y.append(func(val, location))
    plt.plot(x, y)
    plt.savefig(f'{dir}{func.__qualname__}_utility_func.png')
    
beta_test()
PT5_test()
# plt.show()
utility_func_bw_test(800, 8000, UtilityFunc.VoIP.bw_up)
utility_func_bw_test(40, 4000, UtilityFunc.VoIP.bw_down)
utility_func_delay_test(20, 2000, UtilityFunc.VoIP.delay, 'cloud')
utility_func_delay_test(20, 2000, UtilityFunc.VoIP.delay, 'edge')
# plt.show()
utility_func_bw_test(50, 5000, UtilityFunc.IPVideo.bw_up)
utility_func_bw_test(1000, 10000, UtilityFunc.IPVideo.bw_down)
# plt.show()
utility_func_bw_test(50, 5000, UtilityFunc.FTP.bw_up)
utility_func_bw_test(1500, 15000, UtilityFunc.FTP.bw_down)
# plt.show()