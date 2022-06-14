import numpy as np
from time import time
import scipy.integrate as integrate
import math
import logging
from parameters import (rnd_seed, Task_type_index, case_num, big_round_times)
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
    mno_user_cost = [] # float
    mvno_user_cost = [] # float
    mno_cloud_task_num = [] # (VoIP, IP Video, FTP)
    mno_edge_task_num = [] # (VoIP, IP Video, FTP)
    mvno_cloud_task_num = [] # (VoIP, IP Video, FTP)
    mvno_edge_task_num = [] # (VoIP, IP Video, FTP)
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

    @classmethod
    def plot_1dim_line(cls, data):
        x = np.arange(1, len(data) + 1)
        labels = [str(i) for i in x]
        plt.plot(x, data, '-o')

        ax = plt.gca()
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    @classmethod
    def plot_cloud_edge_task_num(cls, data1, data2):
        x = np.arange(1, len(data1) + 1)
        labels = [str(i) for i in x]

        plt.plot(x, data1, 'o-', label='cloud')
        plt.plot(x, data2, 'o-', label='edge')

        ax = plt.gca()
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

    ################# Plotting each type of data #################

    @classmethod
    def plot_statistic_data(cls):
        plt.figure(figsize=cls.figsize)
        # computing resource
        plt.subplot(311)
        plt.title('Statistic data - computing resource in each round')
        plt.xlabel('round')
        plt.ylabel('computing resource (GCUs/s)')
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
        # computing resource
        plt.subplot(311)
        plt.title('Hour data - computing resource in each hour')
        plt.xlabel('hour')
        plt.ylabel('computing resource (GCUs/s)')
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
            # computing resource
            plt.subplot(311)
            plt.title('MNO VM resource - computing resource in each round')
            plt.xlabel('round')
            plt.ylabel('computing resource (GCUs/s)')
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

            plt.savefig(f'figs/{case_num}MNO/mno_vm_resource')

        def plot_task_fitness():
            for day in range(big_round_times - 2):
                plt.figure(figsize=cls.figsize)
                plt.title(f'MNO Task fitness in busy hour - day {day + 1}')
                plt.xlabel('hour')
                plt.ylabel('total fitness in an hour')

                x = np.arange(1, 11) # 8 AM to 5 PM
                labels = [str(i + 7) for i in x]
                plt.plot(x, cls.mno_task_fitness[24 * day + 7:24 * day + 17, 0], 'o-', label='VoIP')
                plt.plot(x, cls.mno_task_fitness[24 * day + 7:24 * day + 17, 1], 'o-', label='IP Video')
                plt.plot(x, cls.mno_task_fitness[24 * day + 7:24 * day + 17, 2], 'o-', label='FTP')

                ax = plt.gca()
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()
                plt.savefig(f'figs/{case_num}MNO/mno_task_fitness_day_{day + 1}')

        def plot_task_resource():
            plt.figure(figsize=cls.figsize)
            # computing resource
            plt.subplot(311)
            plt.title('MNO Task consuming resource - computing resource in each hour')
            plt.xlabel('hour')
            plt.ylabel('computing resource (GCUs/s)')
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

            plt.savefig(f'figs/{case_num}MNO/mno_task_resource')

        def plot_block_rate():
            plt.figure(figsize=cls.figsize)
            plt.title('MNO block ratio in each hour')
            plt.xlabel('hour')
            plt.ylabel('ratio')
            cls.plot_2dim_line(cls.mno_block_rate)
            plt.savefig(f'figs/{case_num}MNO/mno_task_block_rate')

        def plot_user_cost():
            plt.figure(figsize=cls.figsize)
            plt.title('MNO average user cost in each hour')
            plt.xlabel('hour')
            plt.ylabel('cost (dollar)')
            cls.plot_1dim_line(cls.mno_user_cost)
            plt.savefig(f'figs/{case_num}MNO/mno_user_cost')

        def plot_task_num():
            plt.figure(figsize=cls.figsize)
            # VoIP
            plt.subplot(311)
            plt.title('MNO number of task assign to cloud/edge VM in each hour - VoIP')
            plt.xlabel('hour')
            plt.ylabel('number of task')
            cls.plot_cloud_edge_task_num(cls.mno_cloud_task_num[:, 0], cls.mno_edge_task_num[:, 0])
            # IP Video
            plt.subplot(312)
            plt.title('MNO number of task assign to cloud/edge VM in each hour - IP Video')
            plt.xlabel('hour')
            plt.ylabel('number of task')
            cls.plot_cloud_edge_task_num(cls.mno_cloud_task_num[:, 1], cls.mno_edge_task_num[:, 1])
            # FTP
            plt.subplot(313)
            plt.title('MNO number of task assign to cloud/edge VM in each hour - FTP')
            plt.xlabel('hour')
            plt.ylabel('number of task')
            cls.plot_cloud_edge_task_num(cls.mno_cloud_task_num[:, 2], cls.mno_edge_task_num[:, 2])

            plt.savefig(f'figs/{case_num}MNO/mno_task_num')

        plot_vm_resource()
        plot_task_fitness()
        plot_task_resource()
        plot_block_rate()
        plot_user_cost()
        plot_task_num()

    @classmethod
    def plot_mvno(cls):
        def plot_vm_resource():
            plt.figure(figsize=cls.figsize)
            # computing resource
            plt.subplot(311)
            plt.title('MVNO VM resource - computing resource in each round')
            plt.xlabel('round')
            plt.ylabel('computing resource (GCUs/s)')
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

            plt.savefig(f'figs/{case_num}MVNO/mvno_vm_resource')

        def plot_task_fitness():
            for day in range(big_round_times - 2):
                plt.figure(figsize=cls.figsize)
                plt.title(f'MVNO Task fitness in busy hour - day {day + 1}')
                plt.xlabel('hour')
                plt.ylabel('total fitness in an hour')

                x = np.arange(1, 11) # 7 AM to 5 PM
                labels = [str(i + 7) for i in x]
                plt.plot(x, cls.mvno_task_fitness[24 * day + 7:24 * day + 17, 0], 'o-', label='VoIP')
                plt.plot(x, cls.mvno_task_fitness[24 * day + 7:24 * day + 17, 1], 'o-', label='IP Video')
                plt.plot(x, cls.mvno_task_fitness[24 * day + 7:24 * day + 17, 2], 'o-', label='FTP')

                ax = plt.gca()
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()
                plt.savefig(f'figs/{case_num}MVNO/mvno_task_fitness_day_{day + 1}')

        def plot_task_resource():
            plt.figure(figsize=cls.figsize)
            # computing resource
            plt.subplot(311)
            plt.title('MVNO Task consuming resource - computing resource in each hour')
            plt.xlabel('hour')
            plt.ylabel('computing resource (GCUs/s)')
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

            plt.savefig(f'figs/{case_num}MVNO/mvno_task_resource')

        def plot_vm_cost():
            plt.figure(figsize=cls.figsize)
            plt.title('MVNO VM total cost in each round')
            plt.xlabel('round')
            plt.ylabel('VM total cost(dollar)')

            cls.plot_1dim_bar(cls.mvno_vm_cost)
            plt.savefig(f'figs/{case_num}MVNO/mvno_vm_cost')

        def plot_block_rate():
            plt.figure(figsize=cls.figsize)
            plt.title('MVNO block ratio in each hour')
            plt.xlabel('hour')
            plt.ylabel('ratio')
            cls.plot_2dim_line(cls.mvno_block_rate)
            plt.savefig(f'figs/{case_num}MVNO/mvno_task_block_rate')

        def plot_user_cost():
            plt.figure(figsize=cls.figsize)
            plt.title('MVNO average user cost in each hour')
            plt.xlabel('hour')
            plt.ylabel('cost (dollar)')
            cls.plot_1dim_line(cls.mvno_user_cost)
            plt.savefig(f'figs/{case_num}MVNO/mvno_user_cost')

        def plot_task_num():
            plt.figure(figsize=cls.figsize)
            # VoIP
            plt.subplot(311)
            plt.title('MNO number of task assign to cloud/edge VM in each hour - VoIP')
            plt.xlabel('hour')
            plt.ylabel('number of task')
            cls.plot_cloud_edge_task_num(cls.mvno_cloud_task_num[:, 0], cls.mvno_edge_task_num[:, 0])
            # IP Video
            plt.subplot(312)
            plt.title('MNO number of task assign to cloud/edge VM in each hour - IP Video')
            plt.xlabel('hour')
            plt.ylabel('number of task')
            cls.plot_cloud_edge_task_num(cls.mvno_cloud_task_num[:, 1], cls.mvno_edge_task_num[:, 1])
            # FTP
            plt.subplot(313)
            plt.title('MNO number of task assign to cloud/edge VM in each hour - FTP')
            plt.xlabel('hour')
            plt.ylabel('number of task')
            cls.plot_cloud_edge_task_num(cls.mvno_cloud_task_num[:, 2], cls.mvno_edge_task_num[:, 2])

            plt.savefig(f'figs/{case_num}MVNO/mvno_task_num')

        plot_vm_resource()
        plot_task_fitness()
        plot_task_resource()
        plot_vm_cost()
        plot_block_rate()
        plot_user_cost()
        plot_task_num()
    
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
        cls.mno_user_cost = np.array(cls.mno_user_cost)
        cls.mvno_user_cost = np.array(cls.mvno_user_cost)
        cls.mno_cloud_task_num = np.array(cls.mno_cloud_task_num)
        cls.mno_edge_task_num = np.array(cls.mno_edge_task_num)
        cls.mvno_cloud_task_num = np.array(cls.mvno_cloud_task_num)
        cls.mvno_edge_task_num = np.array(cls.mvno_edge_task_num)

        if not os.path.exists(f'figs/{case_num}'):
            os.makedirs(f'figs/{case_num}')
        cls.plot_statistic_data()
        cls.plot_hour_data()
        if not os.path.exists(f'figs/{case_num}MNO/'):
            os.makedirs(f'figs/{case_num}MNO/')
        cls.plot_mno()
        if not os.path.exists(f'figs/{case_num}MVNO/'):
            os.makedirs(f'figs/{case_num}MVNO/')
        cls.plot_mvno()
        # plt.show()
        logging.info(f'Save figs to ./figs/{case_num}!')
        print(f'Save figs to ./figs/{case_num}!')

        # save metrics data
        if not os.path.exists(f'Metrics/{case_num}'):
            os.makedirs(f'Metrics/{case_num}')
        np.save(f'Metrics/{case_num}statistic_data', cls.statistic_data)
        np.save(f'Metrics/{case_num}hour_data', cls.hour_data)
        np.save(f'Metrics/{case_num}mno_vm_resource', cls.mno_vm_resource)
        np.save(f'Metrics/{case_num}mvno_vm_resource', cls.mvno_vm_resource)
        np.save(f'Metrics/{case_num}mvno_vm_cost', cls.mvno_vm_cost)
        np.save(f'Metrics/{case_num}mno_task_fitness', cls.mno_task_fitness)
        np.save(f'Metrics/{case_num}mno_task_resource', cls.mno_task_resource)
        np.save(f'Metrics/{case_num}mvno_task_fitness', cls.mvno_task_fitness)
        np.save(f'Metrics/{case_num}mvno_task_resource', cls.mvno_task_resource)
        np.save(f'Metrics/{case_num}mno_block_rate', cls.mno_block_rate)
        np.save(f'Metrics/{case_num}mvno_block_rate', cls.mvno_block_rate)
        np.save(f'Metrics/{case_num}mno_user_cost', cls.mno_user_cost)
        np.save(f'Metrics/{case_num}mvno_user_cost', cls.mvno_user_cost)
        np.save(f'Metrics/{case_num}mno_cloud_task_num', cls.mno_cloud_task_num)
        np.save(f'Metrics/{case_num}mno_edge_task_num', cls.mno_edge_task_num)
        np.save(f'Metrics/{case_num}mvno_cloud_task_num', cls.mvno_cloud_task_num)
        np.save(f'Metrics/{case_num}mvno_edge_task_num', cls.mvno_edge_task_num)