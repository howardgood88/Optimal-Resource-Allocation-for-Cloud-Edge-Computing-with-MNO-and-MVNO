import abc
import numpy as np
import math
from contract import Contract
from utils import (toSoftmax, get_TD_populations_log_msg)
from parameters import (offspring_number, Task_type_index, _theta, _lambda, mutate_rate, rnd_seed,
                        _gamma, max_searching_times)
import logging

np.random.seed(rnd_seed)

class GeneticOptimizing(abc.ABC):

    @abc.abstractmethod
    def step(self):
        pass
    
    @abc.abstractmethod
    def selection(self):
        pass

    @abc.abstractmethod
    def crossover(self):
        pass

    @abc.abstractmethod
    def mutation(self):
        pass

class TaskDeploymentParametersOptimizing(GeneticOptimizing):

    def __init__(self, op_bw, op_cr):
        self.best_gamma = np.array(_gamma)
        self.best_op_bw = op_bw
        self.best_op_cr = op_cr

        # Save the populations without softmax
        self.new_populations = np.array([self.initialize_population() for _ in range(offspring_number)])
        self.fitness = [0 for _ in range(offspring_number)]
        self.best_population = np.concatenate((self.best_gamma.flatten(), [self.best_op_bw, self.best_op_cr]))
        self.best_fitness = float('-inf')

    def initialize_population(self):
        new_gamma = np.random.uniform(0, 5, self.best_gamma.size)
        new_op_bw = np.random.uniform(200, 400, 1)
        new_op_cr = np.random.uniform(0, 0.4, 1)
        return np.concatenate((new_gamma, new_op_bw, new_op_cr))

    def update_best_population(self) -> None:
        # update best population
        flag = True
        for idx, (fitness, population) in enumerate(zip(self.fitness, self.new_populations)):
            logging.debug(f'population {idx + 1} {toSoftmax(population)[:-2]} with fitness: {fitness}')
            if fitness > self.best_fitness:
                flag = False
                logging.info(f'better population {idx + 1} found, update best population!')
                self.best_fitness = fitness
                self.best_population = population
                self.best_gamma = [population[0:6], population[6:12], population[12:18]]
                self.best_op_bw, self.best_op_cr = population[-2], population[-1]
        if flag:
            logging.info(f'No better population found, keep origin.')
    
    def step(self) -> None:
        '''Get the next valid offsprings.'''
        if sum(self.fitness) // offspring_number == 0:
            logging.info('data not enough for updating, skip updating.')
            return self.new_populations
        parents = self.selection()
        offsprings = self.crossover(parents)
        self.new_populations = self.mutation(offsprings)
        self.fitness = [0 for _ in range(offspring_number)]

    def selection(self) -> np.array:
        '''Stochastic universal sampling.'''
        def SUS(Population: np.array):
            F = sum(self.fitness)
            N = offspring_number
            P = int(F // N)
            Start = np.random.randint(0, P)
            return np.array([Population[Start + i * P] for i in range(N)])
        wheels = []
        for fitness, population in zip(self.fitness, self.new_populations):
            wheel = np.full((math.ceil(max(0, fitness)), *population.shape), population)
            wheels.append(wheel)
        wheel = np.vstack(wheels)
        wheel = wheel.reshape((-1, *self.new_populations[0].shape))
        parents = SUS(wheel)
        logging.debug(get_TD_populations_log_msg('selected parents', parents))
        return parents

    def crossover(self, parents) -> np.array:
        '''Two-points crossover'''
        points = [np.random.randint(0, len(parents[0])) for _ in range(2)]
        left, right = min(points), max(points)
        logging.debug(f'selected points: ({left}, {right})')
        selected_gene = np.zeros((offspring_number, right - left + 1))
        for idx, parent in enumerate(parents):
            selected_gene[idx] = parent[left:right + 1]
        randomize = np.arange(len(selected_gene))
        np.random.shuffle(randomize)
        logging.debug(f'order after shuffle: {randomize}')
        selected_gene = selected_gene[randomize]
        parents[:, left:right + 1] = selected_gene
        logging.debug(get_TD_populations_log_msg('new offsprings after crossover', parents))
        return parents

    def mutation(self, offsprings) -> np.array:
        for i in range(len(offsprings)):
            mutate = np.random.choice([True, False], offsprings[i].shape, p=[mutate_rate, 1 - mutate_rate])
            _message = f'offspring {i + 1} mutate at {np.arange(*offsprings[i].shape)[mutate]} bit, multiply by'
            for j in range(len(offsprings[i])):
                if mutate[j]:
                    if np.random.random() < 0.5:
                        _message += f' 1.2'
                        offsprings[i, j] = offsprings[i, j] * 1.2
                    else:
                        _message += f' 0.8'
                        offsprings[i, j] = offsprings[i, j] * 0.8
            logging.debug(_message)
        logging.debug(get_TD_populations_log_msg('new offsprings after mutation', offsprings))
        return offsprings