import abc
import numpy as np
from parameters import (offspring_number, Task_type_index, _theta, _lambda, mutate_rate, rnd_seed)
from utils import funcCall
from constract import Contract
import math

np.random.seed(rnd_seed)

class GeneticOptimizing(abc.ABC):
    
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
        self.best_fitness = float('inf')

    @funcCall
    def step(self, statistic_data: np.array):
        if self.new_populations is None:
            self.new_populations = np.array([self.choose_vm(self.candidate_vm_id, statistic_data) for _ in range(offspring_number)], dtype=bool)
            return self.new_populations
        else:
            flag = False
            while not flag:
                parents = self.selection()
                offsprings = self.crossover(parents)
                offsprings = self.mutation(offsprings)
                flag = True
                for offspring in offsprings:
                    if not self.check_condition(offspring, statistic_data):
                        flag = False
            return offsprings

    @funcCall
    def choose_vm(self, candidate_vm_id: np.array, statistic_data: np.array) -> np.array:
        '''Random choose a set of vm that fit the condition.'''
        selected_vm = np.zeros(candidate_vm_id.shape, dtype=bool)
        while not self.check_condition(selected_vm, statistic_data):
            selected_vm = np.random.choice([True, False], candidate_vm_id.shape, p=[0.3, 0.7])
        return selected_vm
    
    @funcCall
    def selection(self):
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
        return SUS(wheel)

    @funcCall
    def crossover(self, parents: np.array):
        points = [np.random.randint(0, len(parents[0])) for _ in range(2)]
        left, right = min(points), max(points)
        selected_gene = np.zeros((offspring_number, right - left + 1))
        for idx, parent in enumerate(parents):
            selected_gene[idx] = parent[left:right + 1]
        np.random.shuffle(selected_gene)
        parents[:, left:right + 1] = selected_gene
        return parents

    @funcCall
    def mutation(self, offsprings: np.array):
        for i in range(len(offsprings)):
            mutate = np.random.choice([True, False], offsprings[i].shape, p=[mutate_rate, 1 - mutate_rate])
            for j in range(len(offsprings[i])):
                if mutate[i]:
                    offsprings[i, j] = np.logical_not(offsprings[i, j])
        return offsprings

    @funcCall
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
        self.parent_population = []
        self.best_population = None
    
    @funcCall
    def selection(self):
        pass

    @funcCall
    def crossover(self):
        pass

    @funcCall
    def mutation(self):
        pass