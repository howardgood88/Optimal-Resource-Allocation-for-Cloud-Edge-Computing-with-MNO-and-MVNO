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