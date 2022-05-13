import numpy as np
from parameters import (rnd_seed, _theta, _lambda, Task_type_index)
from optimizing import VMAssignmentOptimizing
from utils import (printReturn, funcCall, optimizing_times, _lambda)
from constract import Contract
import logging

np.random.seed(rnd_seed)

class VMAssignment:
    '''VM Assignment!!!'''
    def __init__(self, contract: Contract, candidate_vm_id: np.array, statistic_data: np.array, vm_list: dict):
        '''
        Parameters
        ----------
        contract : Contract
            Contract object that supply the upper bound and lower bound of bw and cr.
        candidate_vm_id : np.array
            The candidate vm that MVNO can choose.
        statistic_data : np.array
            The statistic data of history data.
        vm_list : dict
            The dict of vm object.
        '''
        self.optimizing = VMAssignmentOptimizing(contract, candidate_vm_id, statistic_data, vm_list)
        self.candidate_vm_id = candidate_vm_id
        self.vm_list = vm_list
    
    @funcCall
    def run(self) -> None:
        '''Start running VMAssignment algorithm.'''
        for _ in range(optimizing_times):
            new_populations = self.optimizing.step()
            for population in new_populations:
                selected_vm_id = self.candidate_vm_id[population]
                _sum = 0
                for idx, vm_id in enumerate(selected_vm_id):
                    _sum += self.vm_list[vm_id].price
                _sum *= _lambda
                self.optimizing.fitness[idx] = _sum
                if _sum > self.optimizing.best_fitness:
                    self.optimizing.best_fitness = _sum
                    self.optimizing.best_population = population
        return self.candidate_vm_id[np.logical_not(self.optimizing.best_population)], self.candidate_vm_id[self.optimizing.best_population]
        
