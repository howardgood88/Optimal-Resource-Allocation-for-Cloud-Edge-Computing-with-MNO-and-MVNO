from parameters import (case_num, testing)
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 26
})

paths = [f'./Metrics/{case_num[:-1]}_{testing}', f'./baselines/VM Load Balance/Metrics/{case_num[:-1]}_{testing}', f'./baselines/Random/Metrics/{case_num[:-1]}_{testing}']

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
        'mvno_block_rate' : np.load(path + 'mvno_block_rate.npy'), # (VoIP, IP Video, FTP)
        'mno_user_cost' : np.load(path + 'mno_user_cost.npy'), # float
        'mvno_user_cost' : np.load(path + 'mvno_user_cost.npy'), # float
        'mno_cloud_task_num' : np.load(path + 'mno_cloud_task_num.npy'), # (VoIP, IP Video, FTP)
        'mno_edge_task_num' : np.load(path + 'mno_edge_task_num.npy'), # (VoIP, IP Video, FTP)
        'mvno_cloud_task_num' : np.load(path + 'mvno_cloud_task_num.npy'), # (VoIP, IP Video, FTP)
        'mvno_edge_task_num' : np.load(path + 'mvno_edge_task_num.npy'), # (VoIP, IP Video, FTP)
    })

_dir = f'./figs/comparison_{testing}'
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
    if len(data1) > 50:
        labels = ['' for i in range(1, len(x) + 1)]
        # for idx, s in zip(range(12, len(data1) + 12, 24), range(len(data1) // 24)):
        #     labels[idx] = str(s + 1)
        for idx, s in zip(range(24, len(data1), 24), range(len(data1) // 24 - 1)):
            labels[idx] = '|'
    else:
        # tick_x = x
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
    if len(data1) > 50:
        labels = ['' for i in range(1, len(x) + 1)]
        # for idx, s in zip(range(12, len(data1) + 12, 24), range(len(data1) // 24)):
        #     labels[idx] = str(s + 1)
        for idx, s in zip(range(24, len(data1), 24), range(len(data1) // 24 - 1)):
            labels[idx] = '|'
    else:
        # tick_x = x
        labels = [str(i) for i in x]
    plt.plot(x, data1, 'o-', label='VATA')
    plt.plot(x, data2, 'o-', label='VM Load Balance')
    plt.plot(x, data3, 'o-', label='Random')

    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

# def plot_1dim_bar(data):
#     x = np.arange(1, len(data) + 1)
#     labels = [str(i) for i in x]
#     plt.bar(x, data, width=width * 2)

#     ax = plt.gca()
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()

###########################

def plot_3x3(metric, op, plt_title, xlabel, file_title):
    if xlabel == 'round':
        plt_func = plot_2dim_bar
    elif xlabel == 'hour':
        plt_func = plot_2dim_line
    # computing resource
    plt.figure(figsize=figsize)
    ## VoIP
    plt.subplot(311)
    plt.title(f'{plt_title} computing resource - VoIP')
    plt.ylabel('computing resource (GCUs/s)')
    plt_func(data[0][metric][:, 0, 0], data[1][metric][:, 0, 0], data[2][metric][:, 0, 0])
    ## IP Video
    plt.subplot(312)
    plt.title(f'{plt_title} computing resource - IP Video')
    plt.ylabel('computing resource (GCUs/s)')
    plt_func(data[0][metric][:, 1, 0], data[1][metric][:, 1, 0], data[2][metric][:, 1, 0])
    ## FTP
    plt.subplot(313)
    plt.title(f'{plt_title} computing resource - FTP')
    plt.xlabel(xlabel)
    plt.ylabel('computing resource (GCUs/s)')
    plt_func(data[0][metric][:, 2, 0], data[1][metric][:, 2, 0], data[2][metric][:, 2, 0])
    plt.savefig(_dir + op + f'{file_title}_cr')
    # uplink throughput
    plt.figure(figsize=figsize)
    ## VoIP
    plt.subplot(311)
    plt.title(f'{plt_title} uplink throughput - VoIP')
    plt.ylabel('throughput (Kbps)')
    plt_func(data[0][metric][:, 0, 1], data[1][metric][:, 0, 1], data[2][metric][:, 0, 1])
    ## IP Video
    plt.subplot(312)
    plt.title(f'{plt_title} uplink throughput - IP Video')
    plt.ylabel('throughput (Kbps)')
    plt_func(data[0][metric][:, 1, 1], data[1][metric][:, 1, 1], data[2][metric][:, 1, 1])
    ## FTP
    plt.subplot(313)
    plt.title(f'{plt_title} uplink throughput - FTP')
    plt.xlabel(xlabel)
    plt.ylabel('throughput (Kbps)')
    plt_func(data[0][metric][:, 2, 1], data[1][metric][:, 2, 1], data[2][metric][:, 2, 1])
    plt.savefig(_dir + op + f'{file_title}_T_up')
    # downlink throughput
    plt.figure(figsize=figsize)
    ## VoIP
    plt.subplot(311)
    plt.title(f'{plt_title} downlink throughput - VoIP')
    plt.ylabel('throughput (Kbps)')
    plt_func(data[0][metric][:, 0, 2], data[1][metric][:, 0, 2], data[2][metric][:, 0, 2])
    ## IP Video
    plt.subplot(312)
    plt.title(f'{plt_title} downlink throughput - IP Video')
    plt.ylabel('throughput (Kbps)')
    plt_func(data[0][metric][:, 1, 2], data[1][metric][:, 1, 2], data[2][metric][:, 1, 2])
    ## FTP
    plt.subplot(313)
    plt.title(f'{plt_title} downlink throughput - FTP')
    plt.xlabel(xlabel)
    plt.ylabel('throughput (Kbps)')
    plt_func(data[0][metric][:, 2, 2], data[1][metric][:, 2, 2], data[2][metric][:, 2, 2])
    plt.savefig(_dir + op + f'{file_title}_T_down')

def plot_2d(metric, op, plt_title, ylabel, file_title):
    if metric == 'mno_task_fitness' or metric == 'mvno_task_fitness':
        _data1, _data2, _data3 = data[0][metric][:120], data[1][metric][:120], data[2][metric][:120]
    else:
        _data1, _data2, _data3 = data[0][metric], data[1][metric], data[2][metric]
    plt.figure(figsize=figsize)
    # VoIP
    plt.subplot(311)
    plt.title(f'{plt_title} - VoIP')
    plt.ylabel(ylabel)
    plot_2dim_line(_data1[:, 0], _data2[:, 0], _data3[:, 0])
    # IP Video
    plt.subplot(312)
    plt.title(f'{plt_title} - IP Video')
    plt.ylabel(ylabel)
    plot_2dim_line(_data1[:, 1], _data2[:, 1], _data3[:, 1])
    # FTP
    plt.subplot(313)
    plt.title(f'{plt_title} - FTP')
    plt.xlabel('hour')
    plt.ylabel(ylabel)
    plot_2dim_line(_data1[:, 2], _data2[:, 2], _data3[:, 2])
    plt.savefig(_dir + op + file_title)

def plot_1d(metric, fig_title, xlabel, ylabel, file_name):
    plt.figure(figsize=figsize)
    plt.title(fig_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plot_2dim_line(data[0][metric], data[1][metric], data[2][metric])
    plt.savefig(_dir + file_name)

########################## 

def plot_cloud_edge_task_num(metric1, metric2, op):
    def plot(data1, data2):
        x = np.arange(1, len(data1) + 1)
        if len(data1) > 50:
            labels = ['' for i in range(1, len(x) + 1)]
            # for idx, s in zip(range(12, len(data1) + 12, 24), range(len(data1) // 24)):
            #     labels[idx] = str(s + 1)
            for idx, s in zip(range(24, len(data1), 24), range(len(data1) // 24 - 1)):
                labels[idx] = '|'
        else:
            labels = [str(i) for i in x]
        plt.plot(x, data1, 'o-', label='cloud')
        plt.plot(x, data2, 'o-', label='edge')

        ax = plt.gca()
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

    def plot_VATA():
        plt.figure(figsize=figsize)
        # VoIP
        plt.subplot(311)
        plt.title('[VATA] number of tasks assign to cloud/edge VM - VoIP')
        plt.ylabel('number of tasks')
        plot(data[0][metric1][:, 0], data[0][metric2][:, 0])
        # IP Video
        plt.subplot(312)
        plt.title('[VATA] number of tasks assign to cloud/edge VM - IP Video')
        plt.ylabel('number of tasks')
        plot(data[0][metric1][:, 1], data[0][metric2][:, 1])
        # FTP
        plt.subplot(313)
        plt.title('[VATA] number of tasks assign to cloud/edge VM - FTP')
        plt.xlabel('hour')
        plt.ylabel('number of tasks')
        plot(data[0][metric1][:, 2], data[0][metric2][:, 2])

        plt.savefig(_dir + op + 'task_num_VATA')

    def plot_load_balance():
        plt.figure(figsize=figsize)
        # VoIP
        plt.subplot(311)
        plt.title('[VM Load Balance] number of tasks assign to cloud/edge VM - VoIP')
        plt.ylabel('number of tasks')
        plot(data[1][metric1][:, 0], data[1][metric2][:, 0])
        # IP Video
        plt.subplot(312)
        plt.title('[VM Load Balance] number of tasks assign to cloud/edge VM - IP Video')
        plt.ylabel('number of tasks')
        plot(data[1][metric1][:, 1], data[1][metric2][:, 1])
        # FTP
        plt.subplot(313)
        plt.title('[VM Load Balance] number of tasks assign to cloud/edge VM - FTP')
        plt.xlabel('hour')
        plt.ylabel('number of tasks')
        plot(data[1][metric1][:, 2], data[1][metric2][:, 2])

        plt.savefig(_dir + op + 'task_num_vm_load_balance')

    def plot_random():
        plt.figure(figsize=figsize)
        # VoIP
        plt.subplot(311)
        plt.title('[Random] number of tasks assign to cloud/edge VM - VoIP')
        plt.ylabel('number of tasks')
        plot(data[2][metric1][:, 0], data[2][metric2][:, 0])
        # IP Video
        plt.subplot(312)
        plt.title('[Random] number of tasks assign to cloud/edge VM - IP Video')
        plt.ylabel('number of tasks')
        plot(data[2][metric1][:, 1], data[2][metric2][:, 1])
        # FTP
        plt.subplot(313)
        plt.title('[Random] number of tasks assign to cloud/edge VM - FTP')
        plt.xlabel('hour')
        plt.ylabel('number of tasks')
        plot(data[2][metric1][:, 2], data[2][metric2][:, 2])

        plt.savefig(_dir + op + 'task_num_random')

    plot_VATA()
    plot_load_balance()
    plot_random()

##########################

plot_3x3('mno_vm_resource', 'MNO/', 'MNO VM', 'round', 'vm_resource')
plot_3x3('mvno_vm_resource', 'MVNO/', 'MVNO VM', 'round', 'vm_resource')
plot_1d('mvno_vm_cost', 'MVNO VM cost', 'round', 'VM total cost(dollar)', 'MVNO/vm_cost')
plot_2d('mno_task_fitness', 'MNO/', 'MNO task fitness in busy days', 'fitness', 'task_fitness')
plot_3x3('mno_task_resource', 'MNO/', 'MNO task', 'hour', 'task_resource')
plot_2d('mvno_task_fitness', 'MVNO/', 'MVNO task fitness in busy days', 'fitness', 'task_fitness')
plot_3x3('mvno_task_resource', 'MVNO/', 'MVNO task', 'hour', 'task_resource')
plot_2d('mno_block_rate', 'MNO/', 'MNO task blocking ratio', 'blocking ratio', 'task_block_rate')
plot_2d('mvno_block_rate', 'MVNO/', 'MVNO task blocking ratio', 'blocking ratio', 'task_block_rate')
plot_1d('mno_user_cost', 'MNO average user cost', 'hour', 'average user cost(dollar)', 'MNO/user_cost')
plot_1d('mvno_user_cost', 'MVNO average user cost', 'hour', 'average user cost(dollar)', 'MVNO/user_cost')
plot_cloud_edge_task_num('mno_cloud_task_num', 'mno_edge_task_num', 'MNO/')
plot_cloud_edge_task_num('mvno_cloud_task_num', 'mvno_edge_task_num', 'MVNO/')

print(f'Save figs to {_dir}!')