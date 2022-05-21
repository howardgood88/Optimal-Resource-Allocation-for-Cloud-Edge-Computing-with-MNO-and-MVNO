import abc
from vm_assignment import VMAssignment
from task_deployment import TaskDeployment
import numpy as np
from parameters import (_mu, title4)
from utils import (step_logger)
from contract import Contract
import logging

class Network_operator(abc.ABC):
    def __init__(self):
        self.hold_vm_id = None
        self._task_deployment = TaskDeployment()

    def deploy_task(self, task: np.array, vm_list: dict) -> None:
        '''Delegate to class TaskDeployment.'''
        self._task_deployment.deploy(self.hold_vm_id, task, vm_list)

    def release_task(self, task: np.array) -> None:
        '''Delegate to class TaskDeployment.'''
        self._task_deployment.release(task)

    def update_task_deployment_parameters(self) -> None:
        '''Delegate to class TaskDeployment.'''
        with step_logger(f'updating {self.name} parameters', title4, f'Finished updating {self.name} parameters'):
            self._task_deployment.update_parameters()

class MVNO(Network_operator):
    def __init__(self):
        super().__init__()
        self.name = 'MVNO'

class MNO(Network_operator):
    def __init__(self, mvno: MVNO, vm_id_list: list, vm_list: dict):
        super().__init__()
        self.name = 'MNO'
        self.mvno = mvno
        # the id of all vm own by MNO, transform to list because vm_id_list never changes.
        self.total_vm_id = np.array(vm_id_list, dtype=list)
        self.contract = Contract()
        self._vm_assignment = VMAssignment(self.contract, self.total_vm_id, vm_list)

    def vm_assignment(self, statistic_data: np.array, vm_list: dict) -> None:
        '''Calculate average vm bw and delegate to class VMAssignment.'''
        def get_avg_vm_bw():
            '''Calculate the average bw from all user to vm.'''
            for vm in vm_list.values():
                bw_up_sum = 0
                bw_down_sum = 0
                for data in vm.from_user.values():
                    bw_up_sum += data['bw_up']
                    bw_down_sum += data['bw_down']
                vm.avg_bw_up = bw_up_sum / len(vm.from_user)
                vm.avg_bw_down = bw_down_sum / len(vm.from_user)
        get_avg_vm_bw()
        self.hold_vm_id, self.mvno.hold_vm_id = self._vm_assignment.run(statistic_data)
        logging.info(f'mno vm id: {self.hold_vm_id}, mvno vm id: {self.mvno.hold_vm_id}')
        
        # the price mvno sells to its customers
        for id in self.mvno.hold_vm_id:
            vm_list[id].price = vm_list[id].price * _mu