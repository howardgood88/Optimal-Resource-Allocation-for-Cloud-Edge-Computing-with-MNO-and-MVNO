import abc
import numpy as np
from vm_assignment import VMAssignment
from task_deployment import TaskDeployment
from contract import Contract
from utils import (step_logger, get_total_resource, timer, Metrics)
from parameters import (_mu, mno_op_bw, mno_op_cr, mvno_op_bw, mvno_op_cr)
import logging

class Network_operator(abc.ABC):
    def __init__(self):
        self.hold_vm_id = None
        self._task_deployment = TaskDeployment(self.name, self.op_bw, self.op_cr)

    def deploy_task(self, task: np.array, vm_list: dict) -> None:
        '''Delegate to class TaskDeployment.'''
        self._task_deployment.deploy(self.hold_vm_id, task, vm_list)

    def release_task(self, task: np.array) -> None:
        '''Delegate to class TaskDeployment.'''
        self._task_deployment.release(task)

class MVNO(Network_operator):
    def __init__(self):
        self.name = 'MVNO'
        self.op_bw = mvno_op_bw
        self.op_cr = mvno_op_cr
        super().__init__()

class MNO(Network_operator):
    def __init__(self, mvno: MVNO, vm_id_list: list, vm_list: dict):
        self.name = 'MNO'
        self.op_bw = mno_op_bw
        self.op_cr = mno_op_cr
        super().__init__()
        self.mvno = mvno
        # the id of all vm own by MNO, transform to list because vm_id_list never changes.
        self.total_vm_id = np.array(vm_id_list, dtype=list)
        self.contract = Contract()
        self._vm_assignment = None

    @timer
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
        self._vm_assignment = VMAssignment(self.contract, self.total_vm_id, vm_list)
        self.hold_vm_id, self.mvno.hold_vm_id = self._vm_assignment.run(statistic_data)
        # MNO
        mno_resource = get_total_resource(self.hold_vm_id, vm_list)
        logging.info(f'mno vm id: {self.hold_vm_id},\ntotal resource (bw_up, bw_down, cr): {mno_resource}')
        Metrics.mno_vm_resource.append(mno_resource)
        # MVNO
        mvno_resource = get_total_resource(self.mvno.hold_vm_id, vm_list)
        mvno_cost = self._vm_assignment.vm_highest_price - self._vm_assignment.optimizing.best_fitness
        logging.info(f'mvno vm id: {self.mvno.hold_vm_id},\ntotal resource (bw_up, bw_down, cr): {mvno_resource}, '
                        f'cost: {mvno_cost}')
        Metrics.mvno_vm_resource.append(mvno_resource)
        Metrics.mvno_vm_cost.append(mvno_cost)

        logging.info(f'contract: bw_high: {self.contract.bw_high}, bw_low: {self.contract.bw_low}, cr_high: {self.contract.cr_high}, cr_low: {self.contract.cr_low}')
        
        # the price mvno sells to its customers
        for id in self.mvno.hold_vm_id:
            vm_list[id].price = vm_list[id].price * _mu