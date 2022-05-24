from platform import release
import numpy as np
from queue import Queue
from vm import VM
from optimizing import TaskDeploymentParametersOptimizing
from task_handler import Task_handler
from utils import (softmax, toSoftmax, step_logger, get_TD_populations_log_msg)
from parameters import (generated_bw_max, generated_bw_min, title5,
                        generated_delay_cloud_max, generated_delay_cloud_min, generated_delay_edge_max,
                        generated_delay_edge_min, Task_type_index, Task_event_index, Global)
import logging

class UtilityFunc:
    '''Mapping from resource to utility from 0 to 100.'''
    '''TODO'''
    class VoIP:
        @staticmethod
        def bw_up(bw: float) -> float:
            return bw / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def bw_down(bw: float) -> float:
            return bw / (generated_bw_max - generated_bw_min) * 100

        @staticmethod
        def cr(cr: float) -> float:
            return cr * 100

        @staticmethod
        def price(p: float) -> float:
            max_price = 150
            return (max_price - p) / max_price * 100

        @staticmethod
        def delay(d: float, location: str) -> float:
            if location == 'cloud':
                return (generated_delay_cloud_max - d) / (generated_delay_cloud_max - generated_delay_cloud_min) * 100
            elif location == 'edge':
                return (generated_delay_edge_max - d) / (generated_delay_edge_max - generated_delay_edge_min) * 100

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
            return cr * 100

        @staticmethod
        def price(p: float) -> float:
            max_price = 150
            return (max_price - p) / max_price * 100

        @staticmethod
        def delay(d: float, location: str) -> float:
            if location == 'cloud':
                return (generated_delay_cloud_max - d) / (generated_delay_cloud_max - generated_delay_cloud_min) * 100
            elif location == 'edge':
                return (generated_delay_edge_max - d) / (generated_delay_edge_max - generated_delay_edge_min) * 100

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
            return cr * 100

        @staticmethod
        def price(p: float) -> float:
            max_price = 150
            return (max_price - p) / max_price * 100

        @staticmethod
        def delay(d: float, location: str) -> float:
            if location == 'cloud':
                return (generated_delay_cloud_max - d) / (generated_delay_cloud_max - generated_delay_cloud_min) * 100
            elif location == 'edge':
                return (generated_delay_edge_max - d) / (generated_delay_edge_max - generated_delay_edge_min) * 100

        @staticmethod
        def cr_diff(diff: float) -> float:
            _range = generated_bw_max - generated_bw_min
            return (_range - diff) / (_range) * 100

    @classmethod
    def get_task_utility_func(cls, task_type: str):
        '''Get task utility class by task name.'''
        if task_type == 'VoIP':
            task_utilities = cls.VoIP
        elif task_type == 'IP_Video':
            task_utilities = cls.IPVideo
        else:
            task_utilities = cls.FTP
        return task_utilities

class TaskDeployment:
    '''Task Deployment!!!'''
    def __init__(self):
        self.optimizing = TaskDeploymentParametersOptimizing()
        # each element with two inner element: start event and end event of a task
        self.unaccepted_task_queue = Queue()
        # the sum of utility in an hour
        self.hour_utility = 0
        # the number of valid task in an hour
        self.hour_task_num = 0
        # the average of hour_utility
        self.hour_fitness = 0
        # keep the mapping of task id to vm it runs
        self.running_task_id_to_vm = {}

    def __enter__(self):
        '''Initialization.'''
        self.hour_utility = 0
        self.hour_task_num = 0

    def __exit__(self, type, value, traceback):
        '''Get the statistic fitness of optimizing populations at the end of an hour.'''
        if self.hour_task_num == 0:
            self.hour_task_num = 1
        # average the utility
        self.hour_fitness = self.hour_utility / self.hour_task_num
        for idx in range(len(self.optimizing.fitness)):
            self.optimizing.fitness[idx] /= self.hour_task_num
        logging.info(f'Release undone tasks: {self.running_task_id_to_vm.keys()}')
        self.all_release()

    def deploy(self, candidate_vm_id: np.array, task: np.array, vm_list: dict) -> None:
        '''Start running TaskDeployment algorithm.'''
        self.hour_task_num += 1
        # get index in task_events.json
        task_type = task[Task_event_index.task_type.value]
        user_id = task[Task_event_index.user_id.value]
        cpu_request = task[Task_event_index.cpu_request.value]

        max_utility = float('-inf')
        offsprings_max_utility = [float('-inf') for _ in range(len(self.optimizing.new_populations))]
        selected_vm_id = None
        for vm_id in candidate_vm_id:
            # deployment by best population
            vm = vm_list[vm_id]
            ## only accept vm of the same type
            if vm.task_type != task_type:
                continue
            ## calculate the utilities
            task_utility = UtilityFunc.get_task_utility_func(task_type)
            bw_up = vm.from_user[user_id]['bw_up']
            bw_down = vm.from_user[user_id]['bw_down']
            delay = vm.from_user[user_id]['delay']
            cr_diff = abs(cpu_request - vm.cr)
            # checking the operating value and vm remaining resource
            if min(bw_up, bw_down) < self.optimizing.best_op_bw or vm.cr < self.optimizing.best_op_cr or \
                vm.cr < task[Task_event_index.average_cpu_usage] or vm.local_bw_up < task[Task_event_index.T_up] or \
                vm.local_bw_down < task[Task_event_index.T_down]:
                continue
            utilities = [
                task_utility.bw_up(bw_up),
                task_utility.bw_down(bw_down),
                task_utility.cr(vm.cr),
                task_utility.price(vm.price),
                task_utility.delay(delay, vm.location),
                task_utility.cr_diff(cr_diff)
            ]
            utility = sum([g * u for g, u in zip(softmax(self.optimizing.best_gamma[Task_type_index[task_type]]), utilities)])
            # virtual deployment by offsprings
            for idx, population in enumerate(self.optimizing.new_populations):
                population = toSoftmax(population)
                _op_bw, _op_cr = population[-2], population[-1]
                if min(bw_up, bw_down) < _op_bw and vm.cr < _op_cr:
                    continue
                _gamma = [population[0:6], population[6:12], population[12:18]]
                _utility = sum([g * u for g, u in zip(_gamma[Task_type_index[task_type]], utilities)])

                if _utility > offsprings_max_utility[idx]:
                    offsprings_max_utility[idx] = _utility
            # keep the best vm
            if utility > max_utility:
                max_utility = utility
                selected_vm_id = vm_id

        # if no feasible solution
        if selected_vm_id == None:
            self.reschedule_task(task)
            self.hour_task_num -= 1
        else:
            self.bind_task(task, vm_list[selected_vm_id])

        self.hour_utility = max(self.hour_utility, max_utility)
        for idx, _utility in enumerate(offsprings_max_utility):
            self.optimizing.fitness[idx] = max(0, _utility)

    def reschedule_task(self, task: np.array) -> None:
        task_id = task[Task_event_index.index.value]
        Task_handler.set_mask(task_id)
        events = Task_handler.get_deleted_events()
        start_event, end_event = events
        assert(len(events) == 2)
        Task_handler.delete_events()
        event_time_idx = Task_event_index.event_time.value
        retry_offset = np.random.randint(60, 120)
        logging.info(f'Task{task_id} unaccepted, retry after {retry_offset} minutes.')
        start_event[event_time_idx] = start_event[event_time_idx] + retry_offset
        end_event[event_time_idx] = end_event[event_time_idx] + retry_offset
        Task_handler.insert_event(end_event)
        Task_handler.insert_event(start_event)

    def bind_task(self, task: np.array, selected_vm: VM) -> None:
        '''Consume resource of selected vm and make task as observer.'''
        task_id = task[Task_event_index.index.value]
        _message = f'Deploy task{task_id} to vm {selected_vm.id},\n'

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

        self.running_task_id_to_vm[task_id] = selected_vm

    def release(self, task: np.array) -> None:
        '''Release vm resource used by task.'''
        task_id = task[Task_event_index.index.value]
        vm = self.running_task_id_to_vm[task_id]

        _message = f'release task{task_id} from vm {vm.id},\n'
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

        del self.running_task_id_to_vm[task_id]

    def update_parameters(self) -> None:
        '''Update the parameters based on the performance of optimizing offsprings of this hour.'''
        with step_logger('Updating best population', title5, f'Finished updating best population.'):
            self.optimizing.best_fitness = self.hour_fitness
            self.optimizing.update_best_population()
            logging.info(f'best population as {toSoftmax(self.optimizing.best_population)}, fitness: {self.optimizing.best_fitness}.')
        with step_logger('Generate new offsprings', title5, 'Finished generating new offsprings.'):
            self.optimizing.step()
            logging.info(get_TD_populations_log_msg('final new offspring', self.optimizing.new_populations))

    def all_release(self):
        tasks = []
        # get and reschedule undone tasks
        for task_id in self.running_task_id_to_vm:
            task_id_idx = Task_event_index.index.value
            task = Task_handler.task_events[Task_handler.task_events[:, task_id_idx] == task_id][0]
            tasks.append(task)
            self.reschedule_task(task)
        # release undone tasks
        for task in tasks:
            self.release(task)
