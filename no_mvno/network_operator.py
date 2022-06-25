import abc
import numpy as np
from task_deployment import TaskDeployment
from parameters import (mno_op_bw, mno_op_cr)

class Network_operator(abc.ABC):
    def __init__(self):
        self.hold_vm_id = None
        self._task_deployment = TaskDeployment(self.name, self.op_bw, self.op_cr)
        self.profit = 0

    def deploy_task(self, task: np.array, vm_list: dict) -> None:
        '''Delegate to class TaskDeployment.'''
        self._task_deployment.deploy(self.hold_vm_id, task, vm_list)

    def release_task(self, task: np.array) -> None:
        '''Delegate to class TaskDeployment.'''
        self._task_deployment.release(task)

class MNO(Network_operator):
    def __init__(self, vm_id_list: list):
        self.name = 'MNO'
        self.op_bw = mno_op_bw
        self.op_cr = mno_op_cr
        super().__init__()
        # the id of all vm own by MNO, transform to list because vm_id_list never changes.
        self.total_vm_id = np.array(vm_id_list, dtype=list)
        self.hold_vm_id = self.total_vm_id