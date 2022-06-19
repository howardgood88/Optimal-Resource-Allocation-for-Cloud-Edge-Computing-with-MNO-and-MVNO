from platform import release
import numpy as np
from queue import Queue
from vm import VM
from task_handler import Task_handler
import math
from utils import (softmax, toSoftmax, step_logger, get_TD_populations_log_msg, sgn, Metrics)
from parameters import *
import logging

np.random.seed(rnd_seed)

class UtilityFunc:
    '''Mapping from resource to utility from 0 to 100.'''
    '''TODO'''
    class VoIP:
        @staticmethod
        def bw_up(bw: float) -> float:
            return max_score * (sgn(bw - voip_bw_up_bmin) + 1) / 2

        @staticmethod
        def bw_down(bw: float) -> float:
            return max_score * (sgn(bw - voip_bw_down_bmin) + 1) / 2

        @staticmethod
        def cr(cr: float) -> float:
            return max_score * cr

        @staticmethod
        def price(p: float) -> float:
            max_price = 250
            return max_score * (max_price - p) / max_price

        @staticmethod
        def delay(d: float) -> float:
            return max_score * delay_factor ** -d

        @staticmethod
        def cr_diff(diff: float) -> float:
            return max_score * (1 - diff)

    class IPVideo:
        @staticmethod
        def bw_up(bw: float) -> float:
            bw = min(bw, ftp_bw_up_bmax)
            return max_score * math.log10(bw / ipVideo_bw_up_bmin) / math.log10(ipVideo_bw_up_bmax / ipVideo_bw_up_bmin) * (sgn(bw - ipVideo_bw_up_bmin) + 1) / 2

        @staticmethod
        def bw_down(bw: float) -> float:
            bw = min(bw, ftp_bw_down_bmax)
            return max_score * math.log10(bw / ipVideo_bw_down_bmin) / math.log10(ipVideo_bw_down_bmax / ipVideo_bw_down_bmin) * (sgn(bw - ipVideo_bw_down_bmin) + 1) / 2

        @staticmethod
        def cr(cr: float) -> float:
            return max_score * cr

        @staticmethod
        def price(p: float) -> float:
            max_price = 250
            return max_score * (max_price - p) / max_price

        @staticmethod
        def delay(d: float) -> float:
            return max_score * delay_factor ** -d

        @staticmethod
        def cr_diff(diff: float) -> float:
            return max_score * (1 - diff)

    class FTP:
        @staticmethod
        def bw_up(bw: float) -> float:
            bw = min(bw, ftp_bw_up_bmax)
            return max_score * math.log10(bw + 1) / math.log10(ftp_bw_up_bmax + 1)

        @staticmethod
        def bw_down(bw: float) -> float:
            bw = min(bw, ftp_bw_down_bmax)
            return max_score * math.log10(bw + 1) / math.log10(ftp_bw_down_bmax + 1)

        @staticmethod
        def cr(cr: float) -> float:
            return cr * max_score

        @staticmethod
        def price(p: float) -> float:
            max_price = 250
            return (max_price - p) / max_price * max_score

        @staticmethod
        def delay(d: float) -> float:
            return max_score * delay_factor ** -d

        @staticmethod
        def cr_diff(diff: float) -> float:
            return max_score * (1 - diff)

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
    def __init__(self, operator, op_bw, op_cr):
        self.op_bw = op_bw
        self.op_cr = op_cr
        self.operator = operator
        # each element with two inner element: start event and end event of a task
        self.unaccepted_task_queue = Queue()
        # the sum of utility in an hour
        self.hour_utility = [0, 0, 0] # VoIP, IP Video, FTP
        # the number of valid task in an hour
        self.hour_task_num = [0, 0, 0] # VoIP, IP Video, FTP
        # block rate of tasks in an hour (exclude the tasks drop cause by the transformation of hour.)
        self.block_num = [0, 0, 0] # VoIP, IP Video, FTP
        # the sum of task resource
        self.hour_task_resource = [[0, 0, 0] for _ in range(3)] # 3x3
        # the average of hour_utility
        self.hour_fitness = [0, 0, 0] # VoIP, IP Video, FTP
        # keep the mapping of task id to vm it runs
        self.running_task_id_to_vm = {}
        # keep the starting time of an hour
        self.starting_systime = None
        # user cost
        self.user_cost = 0
        # number of tasks assign to cloud/edge
        self.hour_cloud_task_num = [0, 0, 0]
        self.hour_edge_task_num = [0, 0, 0]
        # avoid a task to retry too many times in an hour
        self.retry_times = {}

    def __enter__(self):
        '''Initialization.'''
        self.starting_systime = Global.system_time
        assert(self.unaccepted_task_queue.empty())
        self.hour_utility = [0, 0, 0]
        self.population_hour_utility = [[0, 0, 0] for _ in range(offspring_number)]
        self.hour_task_num = [0, 0, 0]
        self.block_num = [0, 0, 0]
        self.hour_task_resource = [[0, 0, 0] for _ in range(3)]
        self.hour_fitness = [0, 0, 0]
        self.population_hour_fitness = [[0, 0, 0] for _ in range(offspring_number)]
        self.user_cost = 0
        assert(len(self.running_task_id_to_vm) == 0)
        self.hour_cloud_task_num = [0, 0, 0]
        self.hour_edge_task_num = [0, 0, 0]
        self.retry_times = {}

    def __exit__(self, type, value, traceback):
        '''Deploy to max resource vm.'''
        for i in range(len(self.hour_task_num)):
            if self.hour_task_num[i] == 0:
                self.hour_task_num[i] = 1
            # average the utility
            self.hour_utility[i] = max(self.hour_utility[i], 0)
            self.hour_fitness[i] = self.hour_utility[i] / self.hour_task_num[i]
            for pop_idx in range(offspring_number):
                self.population_hour_fitness[pop_idx][i] = self.population_hour_utility[pop_idx][i] / self.hour_task_num[i]

        self.all_release()
        if self.operator == 'MNO':
            Metrics.mno_task_fitness.append(self.hour_fitness)
            Metrics.mno_task_resource.append(self.hour_task_resource)
            Metrics.mno_block_rate.append([block_num / (block_num + pass_num) for block_num, pass_num in zip(self.block_num, self.hour_task_num)])
            Metrics.mno_user_cost.append(self.user_cost / sum(self.hour_task_num))
            Metrics.mno_cloud_task_num.append(self.hour_cloud_task_num)
            Metrics.mno_edge_task_num.append(self.hour_edge_task_num)
        else:
            Metrics.mvno_task_fitness.append(self.hour_fitness)
            Metrics.mvno_task_resource.append(self.hour_task_resource)
            Metrics.mvno_block_rate.append([block_num / (block_num + pass_num) for block_num, pass_num in zip(self.block_num, self.hour_task_num)])
            Metrics.mvno_user_cost.append(self.user_cost / sum(self.hour_task_num))
            Metrics.mvno_cloud_task_num.append(self.hour_cloud_task_num)
            Metrics.mvno_edge_task_num.append(self.hour_edge_task_num)

    def deploy(self, candidate_vm_id: np.array, task: np.array, vm_list: dict) -> None:
        '''Start running TaskDeployment algorithm.'''
        # get index in task_events.json
        task_type = task[Task_event_index.task_type.value]
        user_id = task[Task_event_index.user_id.value]
        cpu_request = task[Task_event_index.cpu_request.value]
        task_type_idx = Task_type_index[task_type].value
        
        self.hour_task_num[task_type_idx] += 1
        max_resource = [float('-inf') for _ in range(3)]
        max_utility = float('-inf')
        max_utilities = []
        selected_vm_id = None
        shuffled_candidate_vm_id = candidate_vm_id.copy()
        np.random.shuffle(shuffled_candidate_vm_id)
        cost = 0
        for vm_id in shuffled_candidate_vm_id:
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
            # checking vm remaining resource
            if min(bw_up, bw_down) < self.op_bw or vm.cr < self.op_cr or \
                vm.cr < task[Task_event_index.average_cpu_usage] or vm.avg_bw_up < task[Task_event_index.T_up] or \
                vm.avg_bw_down < task[Task_event_index.T_down]:
                continue
            utilities = [
                task_utility.bw_up(bw_up),
                task_utility.bw_down(bw_down),
                # task_utility.cr(vm.cr),
                task_utility.price(vm.price),
                task_utility.delay(delay)
                # task_utility.cr_diff(cr_diff)
            ]
            utility = sum([g * u for g, u in zip(gamma[Task_type_index[task_type]], 
                utilities)]) / sum(gamma[Task_type_index[task_type]])
            # keep the best vm
            if [vm.cr, bw_up, bw_down] > max_resource:
                max_resource = [vm.cr, bw_up, bw_down]
                max_utility = utility
                max_utilities = utilities
                selected_vm_id = vm_id
                cost = vm.price

        task_id = task[Task_event_index.index.value]
        if task_id not in self.retry_times:
            self.retry_times[task_id] = 0
        if selected_vm_id == None:
            if self.retry_times[task_id] <= 3:
                # if no feasible solution
                self.reschedule_task(task)
                self.retry_times[task_id] += 1
            else:
                Task_handler.set_mask(task_id)
                events = Task_handler.get_deleted_events()
                Task_handler.delete_events()
            self.hour_task_num[task_type_idx] -= 1
            self.block_num[task_type_idx] += 1
        else:
            self.bind_task(task, vm_list[selected_vm_id])
            if vm_list[selected_vm_id].location == 'cloud':
                self.hour_cloud_task_num[task_type_idx] += 1
            else:
                self.hour_edge_task_num[task_type_idx] += 1
        logging.info(f'task utilities: {max_utilities}\n')
        logging.info(f'task utility: {max_utility}\n')
    
        self.hour_utility[task_type_idx] += max(max_utility, -100)
        self.user_cost += cost

    def reschedule_task(self, task: np.array) -> None:
        '''Reschedule event in task_events.'''
        task_id = task[Task_event_index.index.value]
        Task_handler.set_mask(task_id)
        events = Task_handler.get_deleted_events()
        start_event, end_event = events
        assert(len(events) == 2)
        Task_handler.delete_events()
        event_time_idx = Task_event_index.event_time.value
        next_round_start_systime = self.starting_systime + small_round_minutes
        retry_offset = np.random.randint(5, 10)
        if start_event[event_time_idx] + retry_offset < next_round_start_systime and end_event[event_time_idx] + retry_offset >= next_round_start_systime:
            interval = end_event[event_time_idx] - start_event[event_time_idx]
            start_event[event_time_idx] = next_round_start_systime
            end_event[event_time_idx] = next_round_start_systime + interval
            logging.info(f'Task {task_id} unaccepted, retry after {next_round_start_systime - Global.system_time + retry_offset} minutes.')
        else:
            start_event[event_time_idx] = start_event[event_time_idx] + retry_offset
            end_event[event_time_idx] = end_event[event_time_idx] + retry_offset
            logging.info(f'Task {task_id} unaccepted, retry after {retry_offset} minutes.')
        Task_handler.insert_event(end_event)
        Task_handler.insert_event(start_event)

    def bind_task(self, task: np.array, selected_vm: VM) -> None:
        '''Consume resource of selected vm and make task as observer.'''
        task_id = task[Task_event_index.index.value]
        _message = f'Deploy task {task_id} to vm {selected_vm.id},\n'

        task_cr = task[Task_event_index.average_cpu_usage.value]
        _message += f'task cr: {task_cr}, vm cr: {selected_vm.cr} -> '
        selected_vm.cr -= task_cr
        _message += f'{selected_vm.cr}\n'

        task_T_up = task[Task_event_index.T_up.value]
        _message += f'task bw_up: {task_T_up}, avg_bw_up: {selected_vm.avg_bw_up} -> '
        selected_vm.avg_bw_up -= task_T_up
        _message += f'{selected_vm.avg_bw_up}\n'

        task_T_down = task[Task_event_index.T_down.value]
        _message += f'task bw_down: {task_T_down}, avg_bw_down: {selected_vm.avg_bw_down} -> '
        selected_vm.avg_bw_down -= task_T_down
        _message += f'{selected_vm.avg_bw_down}'
        logging.info(_message)
        task_type = task[Task_event_index.task_type.value]
        task_type_idx = Task_type_index[task_type].value
        self.hour_task_resource[task_type_idx] = [sum(i) for i in zip(self.hour_task_resource[task_type_idx], (task_cr, task_T_up, task_T_down))]
       
        self.running_task_id_to_vm[task_id] = selected_vm

    def release(self, task: np.array) -> None:
        '''Release vm resource used by task.'''
        task_id = task[Task_event_index.index.value]
        vm = self.running_task_id_to_vm[task_id]

        _message = f'release task {task_id} from vm {vm.id},\n'
        _message += f'cr: {vm.cr} -> '
        vm.cr += task[Task_event_index.average_cpu_usage.value]
        _message += f'{vm.cr}\n'

        _message += f'avg_bw_up: {vm.avg_bw_up} -> '
        vm.avg_bw_up += task[Task_event_index.T_up.value]
        _message += f'{vm.avg_bw_up}\n'

        _message += f'avg_bw_down: {vm.avg_bw_down} -> '
        vm.avg_bw_down += task[Task_event_index.T_down.value]
        _message += f'{vm.avg_bw_down}\n'
        logging.info(_message)

        del self.running_task_id_to_vm[task_id]

    def all_release(self):
        logging.info(f'Release undone tasks: {self.running_task_id_to_vm.keys()}')
        tasks = []
        # get and reschedule undone tasks
        for task_id in self.running_task_id_to_vm:
            task_id_idx = Task_event_index.index.value
            task = Task_handler.task_events[Task_handler.task_events[:, task_id_idx] == task_id][0]
            tasks.append(task)
            self.reschedule_task(task)
        # release undone tasks
        for task in tasks:
            task_type = task[Task_event_index.task_type.value]
            task_type_idx = Task_type_index[task_type].value
            self.hour_task_num[task_type_idx] -= 1
            self.block_num[task_type_idx] += 1
            self.release(task)