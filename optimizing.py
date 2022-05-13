import abc
import numpy as np
from parameters import (offspring_number, funcCall, Task_type_index, _theta, _lambda)
from constract import Contract

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

    def __init__(self, contract: Contract, candidate_vm_id: np.array, statistic_data: np.array, vm_list: dict):
        self.contract = contract
        self.candidate_vm_id = candidate_vm_id
        self.statistic_data = statistic_data
        self.vm_list = vm_list

        self.new_populations = None
        self.fitness = [0 for i in range(offspring_number)]
        self.best_population = None
        self.best_fitness = float('-inf')

    @funcCall
    def step(self):
        if self.new_populations == None:
            self.new_populations = [self.choose_vm(self.candidate_vm_id) for _ in range(offspring_number)]
            return self.new_populations
        else:
            self.selection()
            self.crossover()
            self.mutation()

    @funcCall
    def choose_vm(self, candidate_vm_id: np.array) -> np.array:
        '''Random choose a set of vm that fit the condition.'''
        selected_vm = np.zeros(candidate_vm_id.shape, dtype=bool)
        while self.check_condition(selected_vm):
            selected_vm = np.random.choice([True, False], candidate_vm_id.shape, p = [0.3, 0.7])
        return selected_vm
    
    @funcCall
    def selection(self):
        pass

    @funcCall
    def crossover(self):
        pass

    @funcCall
    def mutation(self):
        pass

    @funcCall
    def check_condition(self, selected_vm: np.array) -> bool:
        '''Check whether the vm set assign to mvno fit the conditions.'''
        contract = self.contract
        statistic_data = self.statistic_data
        vm_list = self.vm_list
        # get needed value
        ## contract content
        bw_low = contract.bw_low
        bw_high = contract.bw_high
        cr_low = contract.cr_low
        cr_high = contract.cr_high

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
            vm = vm_list[id]
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