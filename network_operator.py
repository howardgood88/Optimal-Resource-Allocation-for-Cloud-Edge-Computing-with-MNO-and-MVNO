import abc
from vm_assignment import VMAssignment
from task_deployment import TaskDeployment
import numpy as np
from parameters import (_mu, Task_event_index)
from utils import (printReturn, funcCall)
from constract import Contract
import logging

class Network_operator(abc.ABC):
    @funcCall
    def redeploy(self, vm_list: dict, system_time: int) -> None:
        '''Redeploy the unaccepted tasks.'''
        queue_size = self._task_deployment.unaccepted_task_queue.qsize()
        while queue_size > 0:
            task = self._task_deployment.unaccepted_task_queue.get()
            duration = task[Task_event_index.end_time.value] - task[Task_event_index.start_time.value]
            task[Task_event_index.start_time.value] = system_time
            task[Task_event_index.end_time.value] = system_time + duration
            self.task_deployment(task, vm_list)
            queue_size -= 1

    @funcCall
    def task_deployment(self, task: np.array, vm_list: dict) -> None:
        '''Delegate to class TaskDeployment.'''
        self._task_deployment.run(self.hold_vm_id, task, vm_list)

class MVNO(Network_operator):
    def __init__(self):
        self.name = 'MVNO'
        self.hold_vm_id = None
        self._task_deployment = TaskDeployment()

class MNO(Network_operator):
    def __init__(self, mvno: MVNO, vm_id_list: list, vm_list: dict):
        self.name = 'MNO'
        self.mvno = mvno
        # the id of all vm own by MNO, transform to list because vm_id_list never changes.
        self.total_vm_id = np.array(vm_id_list, dtype=list)
        self.hold_vm_id = None
        self.contract = Contract()
        self._vm_assignment = VMAssignment(self.contract, self.total_vm_id, vm_list)
        self._task_deployment = TaskDeployment()

    @funcCall
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
        # TODO, update contract?
        get_avg_vm_bw()
        self.hold_vm_id, self.mvno.hold_vm_id = self._vm_assignment.run(statistic_data)
        logging.info(f'mno vm id: {self.hold_vm_id}, mvno vm id: {self.mvno.hold_vm_id}')
        
        # the price mvno sells to its customers
        for id in self.mvno.hold_vm_id:
            vm_list[id].price = vm_list[id].price * _mu