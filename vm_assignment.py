import numpy as np
from optimizing import VMAssignmentOptimizing
from contract import Contract
from utils import (step_logger)
from parameters import (rnd_seed, _lambda, optimizing_times, _lambda, title4)
import logging

class VMAssignment:
    '''VM Assignment!!!'''
    def __init__(self, contract: Contract, candidate_vm_id: np.array, vm_list: dict):
        self.optimizing = VMAssignmentOptimizing(contract, candidate_vm_id, vm_list)
        self.candidate_vm_id = candidate_vm_id
        self.vm_list = vm_list
        self.vm_highest_price = sum([vm.price for vm in vm_list.values()])
    
    def run(self, statistic_data: np.array) -> None:
        '''Start running VM Assignment algorithm.'''
        for i in range(optimizing_times):
            with step_logger(f'-----Evolution {i + 1}-----', title4, f'Finished Evolution {i + 1}', logger=logging.debug):
                new_populations = self.optimizing.step(statistic_data)
                logging.debug(f'best population {self.candidate_vm_id[self.optimizing.best_population]} '\
                    f'with cost {self.vm_highest_price - self.optimizing.best_fitness}, with fitness: {self.optimizing.best_fitness}')
                for idx, population in enumerate(new_populations):
                    selected_vm_id = self.candidate_vm_id[population]
                    cost = 0
                    for vm_id in selected_vm_id:
                        # reverse by highest price value to make optimizing maximize other than minimize.
                        # _lambda is the discount MNO provide to MVNO
                        cost += self.vm_list[vm_id].price * _lambda
                    fitness = self.vm_highest_price - cost
                    logging.debug(f'population {idx + 1} {selected_vm_id} with cost: {cost}, fitness: {fitness}')
                    self.optimizing.fitness[idx] = fitness
                    # save the population with minimum cost
                    if fitness > self.optimizing.best_fitness:
                        self.optimizing.best_fitness = fitness
                        self.optimizing.best_population = population
                        logging.debug(f'better population found, update best population to {selected_vm_id}!')
        return self.candidate_vm_id[np.logical_not(self.optimizing.best_population)], self.candidate_vm_id[self.optimizing.best_population]
        
