from parameters import case_num
import numpy as np
import matplotlib.pyplot as plt
import os

paths = [f'./Metrics/{case_num}', f'./baselines/VM Load Balance/Metrics/{case_num}', f'./baselines/Random/Metrics/{case_num}']

data = []
for path in paths:
    data.append({
        'mno_vm_resource' : np.load(path + 'mno_vm_resource.npy'), # 3x3
        'mvno_vm_resource' : np.load(path + 'mvno_vm_resource.npy'), # 3x3
        'mvno_vm_cost' : np.load(path + 'mvno_vm_cost.npy'), # float
        'mno_task_fitness' : np.load(path + 'mno_task_fitness.npy'), # (VoIP, IP Video, FTP)
        'mno_task_resource' : np.load(path + 'mno_task_resource.npy'), # 3x3
        'mvno_task_fitness' : np.load(path + 'mvno_task_fitness.npy'), # (VoIP, IP Video, FTP)
        'mvno_task_resource' : np.load(path + 'mvno_task_resource.npy'), # 3x3
        'mno_block_rate' : np.load(path + 'mno_block_rate.npy'), # (VoIP, IP Video, FTP)
        'mvno_block_rate' : np.load(path + 'mvno_block_rate.npy') # (VoIP, IP Video, FTP)
    })

_dir = './figs/comparison/'
if not os.path.exists(_dir):
    os.makedirs(_dir)
if not os.path.exists(_dir + 'MNO/'):
    os.makedirs(_dir + 'MNO/')
if not os.path.exists(_dir + 'MVNO/'):
    os.makedirs(_dir + 'MVNO/')

width = 0.2
offset = 0.3
figsize = (16, 12)
def plot_2dim_bar(data1, data2, data3):
    x = np.arange(1, len(data1) + 1)
    labels = [str(i) for i in x]
    plt.bar(x - offset, data1, width=width, label='VATA')
    plt.bar(x, data2, width=width, label='VM Load Balance')
    plt.bar(x + offset, data3, width=width, label='Random')

    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

def plot_2dim_line(data1, data2, data3):
    x = np.arange(1, len(data1) + 1)
    labels = [str(i) for i in x]
    plt.plot(x, data1, 'o-', label='VATA')
    plt.plot(x, data2, 'o-', label='VM Load Balance')
    plt.plot(x, data3, 'o-', label='Random')

    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

def plot_1dim_bar(data):
    x = np.arange(1, len(data) + 1)
    labels = [str(i) for i in x]
    plt.bar(x, data, width=cls.width * 2)

    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

###########################

def plot_3x3(metric, op, plt_title, xlabel, file_title):
    # computing resource
    plt.figure(figsize=figsize)
    ## VoIP
    plt.subplot(311)
    plt.title(f'{plt_title} computing resource - VoIP')
    plt.xlabel(xlabel)
    plt.ylabel('computing resource (GCUs/s)')
    plot_2dim_bar(data[0][metric][:, 0, 0], data[1][metric][:, 0, 0], data[2][metric][:, 0, 0])
    ## IP Video
    plt.subplot(312)
    plt.title(f'{plt_title} computing resource - IP Video')
    plt.xlabel(xlabel)
    plt.ylabel('computing resource (GCUs/s)')
    plot_2dim_bar(data[0][metric][:, 1, 0], data[1][metric][:, 1, 0], data[2][metric][:, 1, 0])
    ## FTP
    plt.subplot(313)
    plt.title(f'{plt_title} computing resource - FTP')
    plt.xlabel(xlabel)
    plt.ylabel('computing resource (GCUs/s)')
    plot_2dim_bar(data[0][metric][:, 2, 0], data[1][metric][:, 2, 0], data[2][metric][:, 2, 0])
    plt.savefig(_dir + op + f'{file_title}_cr')
    # uplink throughput
    plt.figure(figsize=figsize)
    ## VoIP
    plt.subplot(311)
    plt.title(f'{plt_title} uplink throughput - VoIP')
    plt.xlabel(xlabel)
    plt.ylabel('throughput (Kbps)')
    plot_2dim_bar(data[0][metric][:, 0, 1], data[1][metric][:, 0, 1], data[2][metric][:, 0, 1])
    ## IP Video
    plt.subplot(312)
    plt.title(f'{plt_title} uplink throughput - IP Video')
    plt.xlabel(xlabel)
    plt.ylabel('throughput (Kbps)')
    plot_2dim_bar(data[0][metric][:, 1, 1], data[1][metric][:, 1, 1], data[2][metric][:, 1, 1])
    ## FTP
    plt.subplot(313)
    plt.title(f'{plt_title} uplink throughput - FTP')
    plt.xlabel(xlabel)
    plt.ylabel('throughput (Kbps)')
    plot_2dim_bar(data[0][metric][:, 2, 1], data[1][metric][:, 2, 1], data[2][metric][:, 2, 1])
    plt.savefig(_dir + op + f'{file_title}_T_up')
    # downlink throughput
    plt.figure(figsize=figsize)
    ## VoIP
    plt.subplot(311)
    plt.title(f'{plt_title} downlink throughput - VoIP')
    plt.xlabel(xlabel)
    plt.ylabel('throughput (Kbps)')
    plot_2dim_bar(data[0][metric][:, 0, 2], data[1][metric][:, 0, 2], data[2][metric][:, 0, 2])
    ## IP Video
    plt.subplot(312)
    plt.title(f'{plt_title} downlink throughput - IP Video')
    plt.xlabel(xlabel)
    plt.ylabel('throughput (Kbps)')
    plot_2dim_bar(data[0][metric][:, 1, 2], data[1][metric][:, 1, 2], data[2][metric][:, 1, 2])
    ## FTP
    plt.subplot(313)
    plt.title(f'{plt_title} downlink throughput - FTP')
    plt.xlabel(xlabel)
    plt.ylabel('throughput (Kbps)')
    plot_2dim_bar(data[0][metric][:, 2, 2], data[1][metric][:, 2, 2], data[2][metric][:, 2, 2])
    plt.savefig(_dir + op + f'{file_title}_T_down')

def plot_2d(metric, op, plt_title, ylabel, file_title):
    plt.figure(figsize=figsize)
    # VoIP
    plt.subplot(311)
    plt.title(f'{plt_title} - VoIP')
    plt.xlabel('hour')
    plt.ylabel(ylabel)
    plot_2dim_line(data[0][metric][:, 0], data[1][metric][:, 0], data[2][metric][:, 0])
    # IP Video
    plt.subplot(312)
    plt.title(f'{plt_title} - IP Video')
    plt.xlabel('hour')
    plt.ylabel(ylabel)
    plot_2dim_line(data[0][metric][:, 1], data[1][metric][:, 1], data[2][metric][:, 1])
    # FTP
    plt.subplot(313)
    plt.title(f'{plt_title} - FTP')
    plt.xlabel('hour')
    plt.ylabel(ylabel)
    plot_2dim_line(data[0][metric][:, 2], data[1][metric][:, 2], data[2][metric][:, 2])
    plt.savefig(_dir + op + file_title)

##########################

def plot_mvno_vm_cost():
    plt.figure(figsize=figsize)
    plt.title('MVNO VM cost')
    plt.xlabel('round')
    plt.ylabel('VM total cost(dollar)')
    plot_2dim_line(data[0]['mvno_vm_cost'], data[1]['mvno_vm_cost'], data[2]['mvno_vm_cost'])
    plt.savefig('mvno_vm_cost')

plot_3x3('mno_vm_resource', 'MNO/', 'MNO VM', 'round', 'MNO_vm')
plot_3x3('mvno_vm_resource', 'MVNO/', 'MVNO VM', 'round', 'MVNO_vm')
plot_mvno_vm_cost()
plot_2d('mno_task_fitness', 'MNO/', 'MNO task fitness', 'fitness', 'mno_task_fitness')
plot_3x3('mno_task_resource', 'MNO/', 'MNO task', 'hour', 'mno_task')
plot_2d('mvno_task_fitness', 'MVNO/', 'MVNO task fitness', 'fitness', 'mvno_task_fitness')
plot_3x3('mvno_task_resource', 'MVNO/', 'MVNO task', 'hour', 'mvno_task')
plot_2d('mno_block_rate', 'MNO/', 'MNO task block rate', 'block rate(%)', 'mno_task_block_rate')
plot_2d('mvno_block_rate', 'MVNO/', 'MVNO task block rate', 'block rate(%)', 'mvno_task_block_rate')