import numpy as np
from parameters import (rnd_seed, _theta, _lambda, Task_type_index, optimizing_times, _lambda)
from optimizing import VMAssignmentOptimizing
from utils import (printReturn, funcCall)
from constract import Contract
import logging

np.random.seed(rnd_seed)

class VMAssignment:
    '''VM Assignment!!!'''
    def __init__(self, contract: Contract, candidate_vm_id: np.array, vm_list: dict):
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
        self.optimizing = VMAssignmentOptimizing(contract, candidate_vm_id, vm_list)
        self.candidate_vm_id = candidate_vm_id
        self.vm_list = vm_list
        self.vm_highest_price = sum([vm.price for vm in vm_list.values()])
    
    def run(self, statistic_data: np.array) -> None:
        '''Start running VM Assignment algorithm.'''
        for _ in range(optimizing_times):
            logging.info(f'-----Evolution {_ + 1}-----')
            new_populations = self.optimizing.step(statistic_data)
            logging.info(f'best population {self.candidate_vm_id[self.optimizing.best_population]} with cost: {self.optimizing.best_fitness}')
            for idx, population in enumerate(new_populations):
                selected_vm_id = self.candidate_vm_id[population]
                cost = 0
                for vm_id in selected_vm_id:
                    # reverse by highest price value to make optimizing maximize other than minimize.
                    # _lambda is the discount MNO provide to MVNO
                    cost += self.vm_list[vm_id].price * _lambda
                fitness = self.vm_highest_price - cost
                logging.info(f'population {selected_vm_id} with cost: {cost}, fitness: {fitness}')
                self.optimizing.fitness[idx] = fitness
                # save the population with minimum cost
                if fitness > self.optimizing.best_fitness:
                    self.optimizing.best_fitness = fitness
                    self.optimizing.best_population = population
                    logging.info(f'better population found, update best population to {selected_vm_id}!')
        return self.candidate_vm_id[np.logical_not(self.optimizing.best_population)], self.candidate_vm_id[self.optimizing.best_population]
        
