import numpy as np
from time import time
import scipy.integrate as integrate
import math
import logging
from parameters import rnd_seed

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
        msg += f'{idx + 1}: {toSoftmax(population)}\n'
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
    bw_up_sum, bw_down_sum, cr_sum = 0, 0, 0
    for vm_id in vm_id_list:
        vm = vm_list[vm_id]
        bw_up_sum += vm.avg_bw_up
        bw_down_sum += vm.avg_bw_down
        cr_sum += vm.cr
    return bw_up_sum, bw_down_sum, cr_sum

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
    def __init__(self):
        # roundly
        self.statistic_data = [] # 3x3
        self.mno_vm_resource = [] # (bw_up, bw_down, cr)
        self.mvno_vm_resource = [] # (bw_up, bw_down, cr)
        self.mvno_vm_cost = [] # int
        # hourly
        self.task_resource = [] # # (T_up, T_down, cr)
        self.task_fitness = []