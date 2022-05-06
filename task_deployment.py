import numpy as np
from optimizing import GeneticOptimizing
from utils import (printReturn, funcCall)
from queue import Queue
from parameters import (_gamma, _op_bw, _op_cr, generated_bw_max, generated_bw_min, Task_type_index, Task_event_index)
from vm import VM
import logging

class UtilityFunc:
    '''Mapping from resource to utility from 0 to 100.'''
    class VoIP:
        @staticmethod
        def bw_up(bw: float) -> float:
            return bw / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def bw_down(bw: float) -> float:
            return bw / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def cr(cr: float) -> float:
            return cr / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def price(c: float) -> float:
            return c / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def delay(d: float) -> float:
            return d / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def cr_diff(diff: float) -> float:
            _range = generated_bw_max - generated_bw_min
            return (_range - diff) / (_range) * 100

    class IPVideo:
        @staticmethod
        def bw_up(bw: float) -> float:
            return bw / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def bw_down(bw: float) -> float:
            return bw / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def cr(cr: float) -> float:
            return cr / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def price(c: float) -> float:
            return c / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def delay(d: float) -> float:
            return d / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def cr_diff(diff: float) -> float:
            _range = generated_bw_max - generated_bw_min
            return (_range - diff) / (_range) * 100

    class FTP:
        @staticmethod
        def bw_up(bw: float) -> float:
            return bw / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def bw_down(bw: float) -> float:
            return bw / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def cr(cr: float) -> float:
            return cr / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def price(c: float) -> float:
            return c / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def delay(d: float) -> float:
            return d / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def cr_diff(diff: float) -> float:
            _range = generated_bw_max - generated_bw_min
            return (_range - diff) / (_range) * 100

    @classmethod
    def get_task_utility(cls, task_type: str):
        '''Get task utility class by task name.'''
        if task_type == 'VoIP':
            task_utilities = cls.VoIP
        elif task_type == 'IP_Video':
            task_utilities = cls.IPVideo
        else:
            task_utilities = cls.FTP
        return task_utilities

class Runing_task_manager:
    '''Manage the tasks running at vm.'''
    def __init__(self):
        self.observers = Queue()
        self.vm_running_at = Queue()

    def set_time(self, system_time: int) -> None:
        '''Check whether system_time trigger task release.'''
        while not self.observers.empty() and self.observers.queue[0][Task_event_index.end_time.value] < system_time:
            self.release_task()
    
    def release_task(self):
        '''Release vm resource used by task.'''
        task = self.observers.get()
        vm = self.vm_running_at.get()

        _message = f'release task{task[Task_event_index.index.value]} from vm {vm.id},\n'
        _message += f'cr: {vm.cr} -> '
        vm.cr += task[Task_event_index.average_cpu_usage.value]
        _message += f'{vm.cr}\n'

        _message += f'local_bw_up: {vm.local_bw_up} -> '
        vm.local_bw_up += task[Task_event_index.T_up.value]
        _message += f'{vm.local_bw_up}\n'

        _message += f'local_bw_down: {vm.local_bw_down} -> '
        vm.local_bw_down += task[Task_event_index.T_down.value]
        _message += f'{vm.local_bw_down}\n'
        logging.info(_message)

    def bind_task(self, task: np.array, selected_vm: VM):
        '''Consume resource of selected vm and make task as observer.'''
        _message = f'Deploy task{task[Task_event_index.index.value]} to vm {selected_vm.id},\n'
        task_cr = task[Task_event_index.average_cpu_usage]
        _message += f'task cr: {task_cr}, vm cr: {selected_vm.cr} -> '
        selected_vm.cr -= task[Task_event_index.average_cpu_usage.value]
        _message += f'{selected_vm.cr}\n'

        task_T_up = task[Task_event_index.T_up]
        _message += f'task bw_up: {task_T_up}, local_bw_up: {selected_vm.local_bw_up} -> '
        selected_vm.local_bw_up -= task[Task_event_index.T_up.value]
        _message += f'{selected_vm.local_bw_up}\n'

        task_T_down = task[Task_event_index.T_down]
        _message += f'task bw_down: {task_T_down}, local_bw_down: {selected_vm.local_bw_down} -> '
        selected_vm.local_bw_down -= task[Task_event_index.T_down.value]
        _message += f'{selected_vm.local_bw_down}\n'
        logging.info(_message)

        self.observers.put(task)
        self.vm_running_at.put(selected_vm)

class TaskDeployment:
    '''Task Deployment!!!'''
    def __init__(self):
        self.optimizing = GeneticOptimizing() # TODO
        self.unaccepted_task_queue = Queue()
        self.task_manager = Runing_task_manager()

    @funcCall
    def run(self, candidate_vm_id: np.array, task: np.array, vm_list: dict) -> None:
        '''Start running TaskDeployment algorithm.'''
        start_time = task[Task_event_index.start_time.value]
        task_type = task[Task_event_index.task_type.value]
        user_id = task[Task_event_index.user_id.value]
        cpu_request = task[Task_event_index.cpu_request.value]
        # check whether any running tasks need to release
        logging.info(f'system time: {start_time}')
        self.task_manager.set_time(start_time)

        max_utility = float('-inf')
        selected_vm_id = None
        for vm_id in candidate_vm_id:
            vm = vm_list[vm_id]
            if vm.task_type != task_type:
                continue
            # calculate utility
            task_utility = UtilityFunc.get_task_utility(task_type)
            bw_up = vm.from_user[user_id]['bw_up']
            bw_down = vm.from_user[user_id]['bw_down']
            utilities = [
                task_utility.bw_up(bw_up),
                task_utility.bw_down(bw_down),
                task_utility.cr(vm.cr),
                task_utility.price(vm.price),
                task_utility.delay(vm.from_user[user_id]['delay']),
                task_utility.cr_diff(abs(cpu_request - vm.cr))
            ]
            utility = sum([g * u for g, u in zip(_gamma[Task_type_index[task_type]], utilities)])
            if utility > max_utility and min(bw_up, bw_down) >= _op_bw and vm.cr >= _op_cr:
                max_utility = utility
                selected_vm_id = vm_id

        # if no feasible solution
        if selected_vm_id == None:
            logging.info(f'task{task[Task_event_index.index.value]} unaccepted')
            self.unaccepted_task_queue.put(task)
        else:
            self.task_manager.bind_task(task, vm_list[selected_vm_id])