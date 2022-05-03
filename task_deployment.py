import numpy as np
from optimizing import GeneticOptimizing
from utils import (printReturn, funcCall)
from queue import Queue
from parameters import (_gamma, _op_bw, _op_cr, generated_bw_max, generated_bw_min, Task_type)
from vm import VM

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
        while not self.observers.empty() and self.observers.queue[0][2] < system_time:
            self.release_task()
    
    def release_task(self):
        '''Release vm resource task by task.'''
        task = self.observers.get()
        vm = self.vm_running_at.get()
        print(f'release task{task[-1]} from vm {vm.id}, cr: {vm.cr} -> ', end = '')
        vm.cr += task[7]
        print(f'{vm.cr}')

    def bind_task(self, task: np.array, selected_vm: VM):
        '''Consume resource of selected vm and make task as observer.'''
        print(f', vm cr: {selected_vm.cr} -> ', end = '')
        selected_vm.cr -= task[7]
        print(f'{selected_vm.cr}')
        self.observers.put(task)
        self.vm_running_at.put(selected_vm)

class TaskDeployment:
    '''Task Deployment!!!'''
    def __init__(self):
        self.optimizing = GeneticOptimizing() # TODO
        self.unaccepted_task_queue = []
        self.task_manager = Runing_task_manager()

    @funcCall
    def run(self, system_time: int, candidate_vm_id: np.array, input_tasks: np.array, vm_list: dict, unaccepted_mode: bool = False) -> None:
        '''Start running TaskDeployment algorithm.'''
        for task in input_tasks:
            # check whether any running tasks need to release
            self.task_manager.set_time(system_time)

            task_type = task[3]
            user_id = task[4]
            cpu_request = task[6]
            max_utility = float('-inf')
            selected_vm_id = None
            for vm_id in candidate_vm_id:
                vm = vm_list[vm_id]
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
                utility = sum([g * u for g, u in zip(_gamma[Task_type[task_type]], utilities)])
                if utility > max_utility and min(bw_up, bw_down) >= _op_bw and vm.cr >= _op_cr:
                    max_utility = utility
                    selected_vm_id = vm_id

            # if no feasible solution
            if selected_vm_id == None:
                if unaccepted_mode:
                    print(f'task{task[-1]} is dropped because still unaccepted...')
                else:
                    print(f'task{task[-1]} unaccepted...')
                    self.unaccepted_task_queue.append(task)
            else:
                print(f'Deploy task{task[-1]} to vm {selected_vm_id}, ', end = '')
                self.task_manager.bind_task(task, vm_list[selected_vm_id])