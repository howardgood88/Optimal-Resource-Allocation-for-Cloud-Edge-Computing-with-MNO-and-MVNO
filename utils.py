import logging
import numpy as np

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

def print_vm_list(vm_list) -> None:
    logging.debug('[vm info]')
    for vm in vm_list.values():
        logging.debug(f'vm id: {vm.id}\ntask_type: {vm.task_type}\nlocation: {vm.location}\n'\
        f'cr: {vm.cr}\nprice: {vm.price}\nlocal_bw_up: {vm.local_bw_up}\nlocal_bw_down: {vm.local_bw_down}\n'\
        f'from_user: {vm.from_user}\navg_bw_up: {vm.avg_bw_up}\navg_bw_down: {vm.avg_bw_down}')

def toSoftmax(population: np.array) -> np.array:
    return np.concatenate((softmax(population[0:6]), softmax(population[6:12]), softmax(population[12:18]), population[18:]))

def softmax(x: np.array) -> np.array:
    return np.exp(x) / sum(np.exp(x))