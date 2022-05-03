import numpy as np
from parameters import (rnd_seed, _theta, _lambda, Task_type)
from optimizing import GeneticOptimizing
from utils import (printReturn, funcCall)
from constract import Contract

np.random.seed(rnd_seed)

class VMAssignment:
    '''VM Assignment!!!'''
    def __init__(self):
        self.optimizing = GeneticOptimizing() # TODO
    
    @funcCall
    def run(self, contract: Contract, candidate_vm_id: np.array, statistic_data: np.array, vm_list: dict) -> None:
        '''
        Start running VMAssignment algorithm.

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
        # TODO
        while 1:
            selected_vm = self.choose_vm(candidate_vm_id)
            print(f'try candidate vm: {selected_vm}...')
            mvno_vm_id = candidate_vm_id[selected_vm]
            if self.check_condition(mvno_vm_id, vm_list, contract, statistic_data):
                mno_vm_id = candidate_vm_id[np.logical_not(selected_vm)]
                break
            print('candidate vm not legal. try again...')
        return mno_vm_id, mvno_vm_id

    @funcCall
    def choose_vm(self, candidate_vm_id: np.array) -> np.array:
        '''Choose a set of vm.'''
        return np.random.choice([True, False], candidate_vm_id.shape, p = [0.3, 0.7])
        #return np.array([True for i in range(6)])

    @funcCall
    def check_condition(self, selected_vm_id: np.array, vm_list: dict, contract, statistic_data: np.array) -> bool:
        '''Check whether the vm set assign to mvno fit the conditions.'''
        # get needed value
        ## contract content
        bw_low = contract.bw_low
        bw_high = contract.bw_high
        cr_low = contract.cr_low
        cr_high = contract.cr_high

        ## statistic data
        voip_idx = Task_type.VoIP.value
        cr_voip = statistic_data[voip_idx][0]
        T_voip_up = statistic_data[voip_idx][1]
        T_voip_down = statistic_data[voip_idx][2]
        ipVideo_idx = Task_type.IP_Video.value
        cr_ipVideo = statistic_data[ipVideo_idx][0]
        T_ipVideo_up = statistic_data[ipVideo_idx][1]
        T_ipVideo_down = statistic_data[ipVideo_idx][2]
        ftp_idx = Task_type.FTP.value
        cr_ftp = statistic_data[ftp_idx][0]
        T_ftp_up = statistic_data[ftp_idx][1]
        T_ftp_down = statistic_data[ftp_idx][2]

        ## vm bw and cr data of different task type
        bw_up_task_x = [0 for i in range(len(Task_type))]
        bw_down_task_x = [0 for i in range(len(Task_type))]
        cr_task_x = [0 for i in range(len(Task_type))]
        for id in selected_vm_id:
            vm = vm_list[id]
            # the price mvno buy from mno
            price = vm.price * _lambda
            vm_type = vm.task_type
            task_idx = Task_type[vm_type].value
            bw_up_task_x[task_idx] += vm.avg_bw_up
            bw_down_task_x[task_idx] += vm.avg_bw_down
            cr_task_x[task_idx] += vm.cr
        bw_up_sum = sum(bw_up_task_x)
        bw_down_sum = sum(bw_down_task_x)

        # check condition
        return bw_up_task_x[voip_idx] >= T_voip_up and\
            bw_up_task_x[ipVideo_idx] >= T_ipVideo_up and\
            bw_up_task_x[ftp_idx] >= T_ftp_up and\
            bw_down_task_x[voip_idx] >= T_voip_down and\
            bw_down_task_x[ipVideo_idx] >= T_ipVideo_down and\
            bw_down_task_x[ftp_idx] >= T_ftp_down and\
            cr_task_x[voip_idx] >= cr_voip and\
            cr_task_x[ipVideo_idx] >= cr_ipVideo and\
            cr_task_x[ftp_idx] >= cr_ftp and\
            bw_low <= min(bw_up_sum, bw_down_sum) * (1 + _theta) and\
            max(bw_up_sum, bw_down_sum) * (1 + _theta) <= bw_high and\
            cr_low <= sum(cr_task_x) * (1 * _theta) <= cr_high