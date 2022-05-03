import numpy as np
from optimizing import GeneticOptimizing
from utils import (printReturn, funcCall)
import queue
from parameters import (_gamma, _op_bw, _op_cr, generated_bw_max, generated_bw_min, generated_delay_max, generated_delay_min)

class UtilityFunc:
    '''Mapping from resource to utility from 0 to 100.'''
    class VoIP:
        @staticmethod
        def bw_up(bw: float):
            return bw / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def bw_down(bw: float):
            return bw / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def cr(cr: float):
            return cr / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def price(c: float):
            return c / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def delay(d: float):
            return d / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def cr_diff(diff: float):
            return diff / (generated_bw_max - generated_bw_max) * 100

    class IPVideo:
        @staticmethod
        def bw_up(bw: float):
            return bw / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def bw_down(bw: float):
            return bw / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def cr(cr: float):
            return cr / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def price(c: float):
            return c / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def delay(d: float):
            return d / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def cr_diff(diff: float):
            return diff / (generated_bw_max - generated_bw_max) * 100

    class FTP:
        @staticmethod
        def bw_up(bw: float):
            return bw / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def bw_down(bw: float):
            return bw / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def cr(cr: float):
            return cr / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def price(c: float):
            return c / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def delay(d: float):
            return d / (generated_bw_max - generated_bw_max) * 100

        @staticmethod
        def cr_diff(diff: float):
            return diff / (generated_bw_max - generated_bw_max) * 100

    @classmethod
    def get_task_utility(cls, task_type: str):
        if task_type == 'VoIP':
            task_utilities = cls.VoIP
        elif task_type == 'IP_Video':
            task_utilities = cls.IPVideo
        else:
            task_utilities = cls.FTP
        return task_utilities

class TaskDeployment:
    def __init__(self):
        self.optimizing = GeneticOptimizing() # TODO

        # parameters
        self._gamma = _gamma
        self._op_bw = _op_bw
        self._op_cr = _op_cr

    @funcCall
    def run(self, candidate_vm_id: np.array, task_events: np.array, vm_list: dict) -> None:
        unaccepted_queue = queue.Queue()

        for idx, task in enumerate(task_events):
            max_utility = float('-inf')
            task_type = task[3]
            user_id = task[4]
            cpu_request = task[6]
            for vm_id in candidate_vm_id:
                vm = vm_list[vm_id]
                task_utility = UtilityFunc.get_task_utility(task_type)
                utilities = [
                    task_utility.bw_up(vm.from_user[user_id]['bw_up']),
                    task_utility.bw_down(vm.from_user[user_id]['bw_down']),
                    task_utility.cr(vm.cr),
                    task_utility.price(vm.price),
                    task_utility.delay(vm.from_user[user_id]['delay']),
                    task_utility.cr_diff(abs(cpu_request - vm.cr))
                ]
                utility = sum([g * u for g, u in zip(_gamma, utilities)])
                if utility > max_utility:
                    max_utility = utility
                    selected_vm_id = vm_id
            print(f'Deploy task{idx + 1} to vm {selected_vm_id}')
            vm_list[vm_id].running_task.add(idx)