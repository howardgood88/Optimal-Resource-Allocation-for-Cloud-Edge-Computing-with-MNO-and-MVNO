import abc
import numpy as np
from parameters import (offspring_number, Task_type_index, _theta, _lambda, mutate_rate, rnd_seed,
                        _gamma, _op_bw, _op_cr)
from utils import (toSoftmax, get_TD_populations_log_msg)
from contract import Contract
import math
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

class VMAssignmentOptimizing(GeneticOptimizing):

    def __init__(self, contract: Contract, candidate_vm_id: np.array, vm_list: dict):
        self.contract = contract
        self.candidate_vm_id = candidate_vm_id
        self.vm_list = vm_list

        self.new_populations = None
        self.fitness = [0 for i in range(offspring_number)]
        self.best_population = None
        self.best_fitness = float('-inf')
        self.min_population_len = None
        self.valid_evolution_message = None

    def step(self, statistic_data: np.array) -> np.array:
        '''Get the next valid offsprings.'''
        if self.new_populations is None:
            self.new_populations = np.array([self.choose_vm(self.candidate_vm_id, statistic_data) for _ in range(offspring_number)], dtype=bool)
            self.min_population_len = min(len(population) for population in self.new_populations)
            return self.new_populations
        else:
            # to ensure all offsprings fit the conditions
            flag = False
            while not flag:
                # print overall message only if all offsprings fit the conditions
                self.valid_evolution_message = ''
                parents = self.selection()
                offsprings = self.crossover(parents)
                offsprings = self.mutation(offsprings)
                flag = True
                for offspring in offsprings:
                    if not self.check_condition(offspring, statistic_data):
                        flag = False
            logging.debug(self.valid_evolution_message)
            return offsprings

    def choose_vm(self, candidate_vm_id: np.array, statistic_data: np.array) -> np.array:
        '''Random choose a set of vm that fit the condition.'''
        selected_vm = np.zeros(candidate_vm_id.shape, dtype=bool)
        while not self.check_condition(selected_vm, statistic_data):
            selected_vm = np.random.choice([True, False], candidate_vm_id.shape, p=[0.3, 0.7])
        return selected_vm
    
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
            wheel = np.full((math.ceil(fitness), *population.shape), population)
            wheels.append(wheel)
        wheel = np.vstack(wheels)
        wheel = wheel.reshape((-1, *self.new_populations[0].shape))
        parents = SUS(wheel)
        self.valid_evolution_message += f'selected parents:\n{parents}\n'
        return parents

    def crossover(self, parents: np.array) -> np.array:
        '''Two-points crossover'''
        points = [np.random.randint(0, self.min_population_len) for _ in range(2)]
        left, right = min(points), max(points)
        self.valid_evolution_message += f'selected points: ({left}, {right})\n'
        selected_gene = np.zeros((offspring_number, right - left + 1))
        for idx, parent in enumerate(parents):
            selected_gene[idx] = parent[left:right + 1]
        randomize = np.arange(len(selected_gene))
        np.random.shuffle(randomize)
        self.valid_evolution_message += f'order after shuffle: {randomize}, '
        selected_gene = selected_gene[randomize]
        parents[:, left:right + 1] = selected_gene
        self.valid_evolution_message += f'new offsprings after crossover:\n{parents}\n'
        return parents

    def mutation(self, offsprings: np.array) -> np.array:
        for i in range(len(offsprings)):
            mutate = np.random.choice([True, False], offsprings[i].shape, p=[mutate_rate, 1 - mutate_rate])
            self.valid_evolution_message += f'offspring {i + 1} mutate at {np.arange(*offsprings[i].shape)[mutate]} bit\n'
            for j in range(len(offsprings[i])):
                if mutate[j]:
                    offsprings[i, j] = np.logical_not(offsprings[i, j])
        self.valid_evolution_message += f'new offsprings:\n{offsprings}\n'
        return offsprings

    def check_condition(self, selected_vm: np.array, statistic_data: np.array) -> bool:
        '''Check whether the vm set assign to mvno fit the conditions.'''
        # get needed value
        ## contract content
        bw_low = self.contract.bw_low
        bw_high = self.contract.bw_high
        cr_low = self.contract.cr_low
        cr_high = self.contract.cr_high

        ## statistic data
        voip_idx = Task_type_index.VoIP.value
        cr_voip = statistic_data[voip_idx][0]
        T_voip_up = statistic_data[voip_idx][1]
        T_voip_down = statistic_data[voip_idx][2]
        ipVideo_idx = Task_type_index.IP_Video.value
        cr_ipVideo = statistic_data[ipVideo_idx][0]
        T_ipVideo_up = statistic_data[ipVideo_idx][1]
        T_ipVideo_down = statistic_data[ipVideo_idx][2]
        ftp_idx = Task_type_index.FTP.value
        cr_ftp = statistic_data[ftp_idx][0]
        T_ftp_up = statistic_data[ftp_idx][1]
        T_ftp_down = statistic_data[ftp_idx][2]

        ## vm bw and cr data of different task type
        bw_up_task_x = [0 for i in range(len(Task_type_index))]
        bw_down_task_x = [0 for i in range(len(Task_type_index))]
        cr_task_x = [0 for i in range(len(Task_type_index))]
        for id in self.candidate_vm_id[selected_vm]:
            vm = self.vm_list[id]
            # the price mvno buy from mno
            price = vm.price * _lambda
            vm_type = vm.task_type
            task_idx = Task_type_index[vm_type].value
            bw_up_task_x[task_idx] += vm.avg_bw_up
            bw_down_task_x[task_idx] += vm.avg_bw_down
            cr_task_x[task_idx] += vm.cr
        bw_up_sum = sum(bw_up_task_x)
        bw_down_sum = sum(bw_down_task_x)

        # check condition
        bw_up_voip_cond = bw_up_task_x[voip_idx] >= T_voip_up
        bw_up_ipVideo_cond = bw_up_task_x[ipVideo_idx] >= T_ipVideo_up
        bw_up_ftp_cond = bw_up_task_x[ftp_idx] >= T_ftp_up
        bw_down_voip_cond = bw_down_task_x[voip_idx] >= T_voip_down
        bw_down_ipVideo_cond = bw_down_task_x[ipVideo_idx] >= T_ipVideo_down
        bw_down_ftp_cond = bw_down_task_x[ftp_idx] >= T_ftp_down
        cr_voip_cond = cr_task_x[voip_idx] >= cr_voip
        cr_ipVideo_cond = cr_task_x[ipVideo_idx] >= cr_ipVideo
        cr_ftp_cond = cr_task_x[ftp_idx] >= cr_ftp
        bw_low_cond = bw_low <= min(bw_up_sum, bw_down_sum) * (1 + _theta)
        bw_high_cond = max(bw_up_sum, bw_down_sum) * (1 + _theta) <= bw_high
        cr_cond = cr_low <= sum(cr_task_x) * (1 * _theta) <= cr_high

        return bw_up_voip_cond and\
            bw_up_ipVideo_cond and\
            bw_up_ftp_cond and\
            bw_down_voip_cond and\
            bw_down_ipVideo_cond and\
            bw_down_ftp_cond and\
            cr_voip_cond and\
            cr_ipVideo_cond and\
            cr_ftp_cond and\
            bw_low_cond and\
            bw_high_cond and\
            cr_cond

class TaskDeploymentParametersOptimizing(GeneticOptimizing):

    def __init__(self):
        self.best_gamma = np.array(_gamma)
        self.best_op_bw = _op_bw
        self.best_op_cr = _op_cr

        # Save the populations without softmax
        self.new_populations = np.array([self.get_parameters() for _ in range(offspring_number)])
        self.fitness = [0 for _ in range(offspring_number)]
        self.best_population = np.concatenate((self.best_gamma.flatten(), [self.best_op_bw, self.best_op_cr]))
        self.best_fitness = float('-inf')

    def get_parameters(self):
        new_gamma = np.random.uniform(0, 5, self.best_gamma.size)
        new_op_bw = np.random.uniform(200, 400, 1)
        new_op_cr = np.random.uniform(0, 0.4, 1)
        return np.concatenate((new_gamma, new_op_bw, new_op_cr))

    def update_best_population(self) -> None:
        # update best population
        for idx, (fitness, population) in enumerate(zip(self.fitness, self.new_populations)):
            logging.debug(f'population {idx + 1} {toSoftmax(population)} with fitness: {fitness}')
            if fitness > self.best_fitness:
                logging.debug(f'better population found, update best population to {toSoftmax(population)}!')
                self.best_fitness = fitness
                self.best_population = population
                self.best_gamma = [population[0:6], population[6:12], population[12:18]]
                self.best_op_bw, self.best_op_cr = population[-2], population[-1]
    
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
            wheel = np.full((math.ceil(fitness), *population.shape), population)
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