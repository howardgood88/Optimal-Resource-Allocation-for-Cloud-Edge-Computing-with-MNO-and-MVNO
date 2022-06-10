import numpy as np
from time import time
import scipy.integrate as integrate
import math
import logging
from parameters import (rnd_seed, Task_type_index, case_num)
import matplotlib.pyplot as plt
import os

np.random.seed(rnd_seed)

def printReturn(func):
    def decorate(*args, **kwargs):
        data = func(*args, **kwargs)
        logging.debug(f'The return data from {func.__name__} is:')
        logging.debug(f'data: {data}')
        logging.debug('----------------------------------------------------------')
        return data
    return decorate

def funcCall(func):
    def decorate(*args, **kwargs):
        logging.debug(f'Function {func.__name__} is called...')
        data = func(*args, **kwargs)
        return data
    return decorate

def toSoftmax(population: np.array) -> np.array:
    return np.concatenate((softmax(population[0:6]), softmax(population[6:12]), softmax(population[12:18]), population[18:]))

def softmax(x: np.array) -> np.array:
    return np.exp(x) / sum(np.exp(x))

def get_TD_populations_log_msg(msg: str, populations: np.array) -> None:
    ''''Log the Task Deployment populations by toSoftmax.'''
    msg += ':\n'
    for idx, population in enumerate(populations):
        msg += f'{idx + 1}: {toSoftmax(population)[:-2]}\n'
    return msg

class step_logger:
    '''
    INFO level logging of algorithm step that need start and end message.
    Something like:
    ----start of xxx----
    ...
    Finished xxx.
    '''
    def __init__(self, in_msg: str, in_title: int, out_msg: str, logger=logging.info):
        self.in_msg = in_msg
        self.in_title = in_title
        self.out_msg = out_msg
        self.logger = logger

    def __enter__(self):
        self.logger(f'{self.in_msg:-^{self.in_title}}')

    def __exit__(self, type, value, traceback):
        self.logger(self.out_msg)

def get_total_resource(vm_id_list: np.array, vm_list: dict):
    _sum = [[0, 0, 0] for _ in range(3)]
    for vm_id in vm_id_list:
        vm = vm_list[vm_id]
        task_type_idx = Task_type_index[vm.task_type].value
        _sum[task_type_idx][0] += vm.cr
        _sum[task_type_idx][1] += vm.avg_bw_up
        _sum[task_type_idx][2] += vm.avg_bw_down
    return _sum

def timer(func):
    def decorate(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        logging.info(f'Function {func.__name__} executed in {time() - t1:.4f}s')
        return result
    return decorate

def beta(a, b, t, d):
    '''Beta Distribution Generator.'''
    def distribution(x):
        return ((x - d) / t) ** (a - 1) * (1 - (x - d) / t) ** (b - 1) / integrate.quad(lambda x: x ** (a - 1) * (1 - x) ** (b - 1), 0, 1)[0]

    mode = (a - 1) / (a + b - 2)
    max_val = distribution(mode * t + d)
    while True:
        x = np.random.uniform(d, d + t)
        y = np.random.uniform(0, max_val)
        if y <= distribution(x):
            return x * 1000 # to Kbps

def PT5(a, b, d, max_x = 20):
    '''Pearson Type 5 Distribution Generator.'''
    def distribution(x):
        return (x - d) ** -(a - 1) * math.exp(-b / (x - d)) * b ** a / math.factorial(a - 1)

    mode = b / (a + 1) + d
    max_val = distribution(mode)
    while True:
        x = np.random.uniform(d, max_x + d)
        y = np.random.uniform(0, max_val)
        if y <= distribution(x):
            return x # ms

def sgn(x):
    if x >= 0:
        return 1
    else:
        return -1

class Metrics:
    '''For plotting the result.'''
    # roundly
    statistic_data = [] # 3x3
    mno_vm_resource = [] # 3x3
    mvno_vm_resource = [] # 3x3
    mvno_vm_cost = [] # float
    # hourly
    hour_data = [] # 3x3
    mno_task_fitness = [] # (VoIP, IP Video, FTP)
    mno_task_resource = [] # 3x3
    mvno_task_fitness = [] # (VoIP, IP Video, FTP)
    mvno_task_resource = [] # 3x3
    mno_block_rate = [] # (VoIP, IP Video, FTP)
    mvno_block_rate = [] # (VoIP, IP Video, FTP)
    # parameters
    offset = 0.3
    gap = 0.05
    width = 0.2
    figsize = (16, 12)

    ################# Common-used Functions #################

    # @classmethod
    # def plot_3dim_bar(cls, data):
    #     ax1 = plt.gca()
    #     ax2 = ax1.twinx()
    #     x = np.arange(1, len(data) + 1)
    #     labels = [str(i) for i in x]
    #     # VoIP
    #     ax1.bar(x - cls.offset - cls.gap, data[:, 0, 0], width=cls.width/1.5, color='tab:pink', label='VoIP cr(GCUs-s)')
    #     ax2.bar(x - cls.offset + cls.gap, data[:, 0, 1], width=cls.width/1.5, label='VoIP bw up(kbps)')
    #     ax2.bar(x - cls.offset + cls.gap, data[:, 0, 2], width=cls.width/1.5, bottom=data[:, 0, 1], label='VoIP bw down(kbps)')
    #     # IP Video
    #     ax1.bar(x - cls.gap, data[:, 1, 0], width=cls.width/1.5, color='tab:gray', label='IP Video cr(GCUs-s)')
    #     ax2.bar(x + cls.gap, data[:, 1, 1], width=cls.width/1.5, label='IP Video bw up(kbps)')
    #     ax2.bar(x + cls.gap, data[:, 1, 2], width=cls.width/1.5, bottom=data[:, 1, 1], label='IP Video bw down(kbps)')
    #     # FTP
    #     ax1.bar(x + cls.offset - cls.gap, data[:, 2, 0], width=cls.width/1.5, color='tab:olive', label='FTP cr(GCUs-s)')
    #     ax2.bar(x + cls.offset + cls.gap, data[:, 2, 1], width=cls.width/1.5, label='FTP bw up(kbps)')
    #     ax2.bar(x + cls.offset + cls.gap, data[:, 2, 2], width=cls.width/1.5, bottom=data[:, 2, 1], label='FTP bw down(kbps)')

    #     ax1.set_xticks(x)
    #     ax1.set_xticklabels(labels)
    #     ax1.legend(loc='upper right')
    #     ax1.set_ylabel('cr(GCUs/s)')
    #     ax2.set_xticks(x)
    #     ax2.set_xticklabels(labels)
    #     ax2.legend(loc='upper left')
    #     ax2.set_ylabel('bw(Kbps)')

    @classmethod
    def plot_2dim_bar(cls, data):
        x = np.arange(1, len(data) + 1)
        labels = [str(i) for i in x]
        plt.bar(x - cls.offset, data[:, 0], width=cls.width, label='VoIP')
        plt.bar(x, data[:, 1], width=cls.width, label='IP Video')
        plt.bar(x + cls.offset, data[:, 2], width=cls.width, label='FTP')

        ax = plt.gca()
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

    @classmethod
    def plot_2dim_line(cls, data):
        x = np.arange(1, len(data) + 1)
        labels = [str(i) for i in x]
        plt.plot(x, data[:, 0], 'o-', label='VoIP')
        plt.plot(x, data[:, 1], 'o-', label='IP Video')
        plt.plot(x, data[:, 2], 'o-', label='FTP')

        ax = plt.gca()
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

    @classmethod
    def plot_1dim_bar(cls, data):
        x = np.arange(1, len(data) + 1)
        labels = [str(i) for i in x]
        plt.bar(x, data, width=cls.width * 2)

        ax = plt.gca()
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

    ################# Plotting each type of data #################

    @classmethod
    def plot_statistic_data(cls):
        plt.figure(figsize=cls.figsize)
        # cr
        plt.subplot(311)
        plt.title('Statistic data - cr in each round')
        plt.xlabel('round')
        plt.ylabel('cr (GCUs/s)')
        cls.plot_2dim_line(cls.statistic_data[:, :, 0])
        # T up
        plt.subplot(312)
        plt.title('Statistic data - uplink throughput in each round')
        plt.xlabel('round')
        plt.ylabel('throughput (Kbps)')
        cls.plot_2dim_line(cls.statistic_data[:, :, 1])
        # T down
        plt.subplot(313)
        plt.title('Statistic data - downlink throughput in each round')
        plt.xlabel('round')
        plt.ylabel('throughput (Kbps)')
        cls.plot_2dim_line(cls.statistic_data[:, :, 2])

        plt.savefig(f'figs/{case_num}statistic_data')

    @classmethod
    def plot_hour_data(cls):
        plt.figure(figsize=cls.figsize)
        # cr
        plt.subplot(311)
        plt.title('Hour data - cr in each hour')
        plt.xlabel('hour')
        plt.ylabel('cr (GCUs/s)')
        cls.plot_2dim_line(cls.hour_data[:, :, 0])
        # T up
        plt.subplot(312)
        plt.title('Hour data - uplink throughput in each hour')
        plt.xlabel('hour')
        plt.ylabel('throughput (Kbps)')
        cls.plot_2dim_line(cls.hour_data[:, :, 1])
        # T down
        plt.subplot(313)
        plt.title('Hour data - downlink throughput in each hour')
        plt.xlabel('hour')
        plt.ylabel('throughput (Kbps)')
        cls.plot_2dim_line(cls.hour_data[:, :, 2])

        plt.savefig(f'figs/{case_num}hour_data_resource')

    @classmethod
    def plot_mno(cls):
        def plot_vm_resource():
            plt.figure(figsize=cls.figsize)
            # cr
            plt.subplot(311)
            plt.title('MNO VM resource - cr in each round')
            plt.xlabel('round')
            plt.ylabel('cr (GCUs/s)')
            cls.plot_2dim_bar(cls.mno_vm_resource[:, :, 0])
            # T up
            plt.subplot(312)
            plt.title('MNO VM resource - uplink throughput in each round')
            plt.xlabel('round')
            plt.ylabel('average throughput (Kbps)')
            cls.plot_2dim_bar(cls.mno_vm_resource[:, :, 1])
            # T down
            plt.subplot(313)
            plt.title('MNO VM resource - downlink throughput in each round')
            plt.xlabel('round')
            plt.ylabel('average throughput (Kbps)')
            cls.plot_2dim_bar(cls.mno_vm_resource[:, :, 2])

            plt.savefig(f'figs/{case_num}mno_vm_resource')

        def plot_task_fitness():
            plt.figure(figsize=cls.figsize)
            plt.title('MNO Task fitness in each hour')
            plt.xlabel('hour')
            plt.ylabel('total fitness in an hour')
            cls.plot_2dim_line(cls.mno_task_fitness)
            plt.savefig(f'figs/{case_num}mno_task_fitness')

        def plot_task_resource():
            plt.figure(figsize=cls.figsize)
            # cr
            plt.subplot(311)
            plt.title('MNO Task consuming resource - cr in each hour')
            plt.xlabel('hour')
            plt.ylabel('cr (GCUs/s)')
            cls.plot_2dim_line(cls.mno_task_resource[:, :, 0])
            # T up
            plt.subplot(312)
            plt.title('MNO Task consuming resource - uplink throughput in each hour')
            plt.xlabel('hour')
            plt.ylabel('average throughput (Kbps)')
            cls.plot_2dim_line(cls.mno_task_resource[:, :, 1])
            # T up
            plt.subplot(313)
            plt.title('MNO Task consuming resource - downlink throughput in each hour')
            plt.xlabel('hour')
            plt.ylabel('average throughput (Kbps)')
            cls.plot_2dim_line(cls.mno_task_resource[:, :, 2])

            plt.savefig(f'figs/{case_num}mno_task_resource')

        plot_vm_resource()
        plot_task_fitness()
        plot_task_resource()

    @classmethod
    def plot_mno_block_rate(cls):
        plt.figure(figsize=cls.figsize)
        plt.title('MNO block rate in each hour')
        plt.xlabel('hour')
        plt.ylabel('percentage (%)')
        cls.plot_2dim_line(cls.mno_block_rate)
        plt.savefig(f'figs/{case_num}mno_task_block_rate')

    @classmethod
    def plot_mvno(cls):
        def plot_vm_resource():
            plt.figure(figsize=cls.figsize)
            # cr
            plt.subplot(311)
            plt.title('MVNO VM resource - cr in each round')
            plt.xlabel('round')
            plt.ylabel('cr (GCUs/s)')
            cls.plot_2dim_bar(cls.mvno_vm_resource[:, :, 0])
            # T up
            plt.subplot(312)
            plt.title('MVNO VM resource - uplink throughput in each round')
            plt.xlabel('round')
            plt.ylabel('average throughput (Kbps)')
            cls.plot_2dim_bar(cls.mvno_vm_resource[:, :, 1])
            # T down
            plt.subplot(313)
            plt.title('MVNO VM resource - downlink throughput in each round')
            plt.xlabel('round')
            plt.ylabel('average throughput (Kbps)')
            cls.plot_2dim_bar(cls.mvno_vm_resource[:, :, 2])

            plt.savefig(f'figs/{case_num}mvno_vm_resource')

        def plot_task_fitness():
            plt.figure(figsize=cls.figsize)
            plt.title('MVNO Task fitness in each hour')
            plt.xlabel('hour')
            plt.ylabel('total fitness in an hour')
            cls.plot_2dim_line(cls.mvno_task_fitness)
            plt.savefig(f'figs/{case_num}mvno_task_fitness')

        def plot_task_resource():
            plt.figure(figsize=cls.figsize)
            # cr
            plt.subplot(311)
            plt.title('MVNO Task consuming resource - cr in each hour')
            plt.xlabel('hour')
            plt.ylabel('cr (GCUs/s)')
            cls.plot_2dim_line(cls.mvno_task_resource[:, :, 0])
            # T up
            plt.subplot(312)
            plt.title('MVNO Task consuming resource - uplink throughput in each hour')
            plt.xlabel('hour')
            plt.ylabel('average throughput (Kbps)')
            cls.plot_2dim_line(cls.mvno_task_resource[:, :, 1])
            # T up
            plt.subplot(313)
            plt.title('MVNO Task consuming resource - downlink throughput in each hour')
            plt.xlabel('hour')
            plt.ylabel('average throughput (Kbps)')
            cls.plot_2dim_line(cls.mvno_task_resource[:, :, 2])

            plt.savefig(f'figs/{case_num}mvno_task_resource')

        plot_vm_resource()
        plot_task_fitness()
        plot_task_resource()

    @classmethod
    def plot_mvno_vm_cost(cls):
        plt.figure(figsize=cls.figsize)
        plt.title('MVNO VM total cost in each round')
        plt.xlabel('round')
        plt.ylabel('VM total cost(dollar)')

        cls.plot_1dim_bar(cls.mvno_vm_cost)
        plt.savefig(f'figs/{case_num}mvno_vm_cost')

    @classmethod
    def plot_mvno_block_rate(cls):
        plt.figure(figsize=cls.figsize)
        plt.title('MVNO block rate in each hour')
        plt.xlabel('hour')
        plt.ylabel('percentage (%)')
        cls.plot_2dim_line(cls.mvno_block_rate)
        plt.savefig(f'figs/{case_num}mvno_task_block_rate')
    
    @classmethod
    def plot(cls):
        cls.statistic_data = np.array(cls.statistic_data)
        cls.hour_data = np.array(cls.hour_data)
        cls.mno_vm_resource = np.array(cls.mno_vm_resource)
        cls.mvno_vm_resource = np.array(cls.mvno_vm_resource)
        cls.mvno_vm_cost = np.array(cls.mvno_vm_cost)
        cls.mno_task_fitness = np.array(cls.mno_task_fitness)
        cls.mno_task_resource = np.array(cls.mno_task_resource)
        cls.mvno_task_fitness = np.array(cls.mvno_task_fitness)
        cls.mvno_task_resource = np.array(cls.mvno_task_resource)
        cls.mno_block_rate = np.array(cls.mno_block_rate)
        cls.mvno_block_rate = np.array(cls.mvno_block_rate)

        if not os.path.exists(f'figs/{case_num}'):
            os.makedirs(f'figs/{case_num}')
        cls.plot_statistic_data()
        cls.plot_hour_data()
        cls.plot_mno()
        cls.plot_mno_block_rate()
        cls.plot_mvno()
        cls.plot_mvno_vm_cost()
        cls.plot_mvno_block_rate()
        # plt.show()
        logging.info(f'Save figs to ./figs/{case_num}!')
        print(f'Save figs to ./figs/{case_num}!')