import json
import numpy as np
from utils import (printReturn, funcCall)
from network_operator import (MNO, MVNO)
from parameters import (small_round_minutes, big_round_minutes, rnd_seed, Task_type, _alpha,
                        generated_bw_max, generated_bw_min, generated_delay_max, generated_delay_min)
from vm import VM

np.random.seed(rnd_seed)

@funcCall
def load_task_data(filename: str) -> np.array:
    '''Load task_events.json and history_data.json from filename'''
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data, dtype=list)

@funcCall
def load_machine_data(filename: str) -> dict:
    '''Load machine_attributes.json from filename'''
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

@funcCall
def data_preprocessing(history_data: np.array, system_time: int) -> tuple[np.array, np.array, set]:
    '''
    Transform task-level history data into hourly history data, and generate user list.
    For N = number of tasks, H = number of hours.

    Parameters
    ----------
    history_data : np.array, shape: (N, 10)
        Task-level history data.
    system_time : int
        The descrete system time.
    
    Returns
    ----------
    hourly_history_data : np.array, shape: (H,3,3)
        The hourly average history data.
    hourly_user_list : np.array[set], shape: (H, *)
        The hourly appeared user list. Each hour with a set.
    user_list : set
        All user appeared in history data.
    '''
    # each row with hourly VoIP, IP_Video, FTP data
    hourly_history_data = []
    # each row with the set of user appeared in the hour
    hourly_user_id_list = []
    # all user had appeared
    user_id_list = set()
    # the selected time range
    minutes_range = (system_time, system_time + small_round_minutes)
    while history_data[history_data[:, 1] > minutes_range[0]].size != 0:
        # hourly mask that fit the minutes_range
        hour_mask = (minutes_range[0] <= history_data[:, 1]) & (history_data[:, 1] < minutes_range[1])

        # calculate hourly average_cpu_usage, bw_up, bw_down of different task from history data
        hourly_tasks_data = [None for i in range(len(Task_type))]
        for task_type, task_idx in Task_type.__members__.items():
            # [average_cpu, bw_up, bw_down]
            data = history_data[hour_mask & (history_data[:, 3] == task_type)][:, -3:]
            hourly_data = np.zeros(3) if data.size == 0 else np.mean(data, axis=0)
            hourly_tasks_data[task_idx.value] = hourly_data
        hourly_history_data.append(hourly_tasks_data)

        # user appeared in this hour
        users = set(history_data[hour_mask][:, 4])
        hourly_user_id_list.append(users)

        # user had appeared
        user_id_list = user_id_list | users

        # update minutes_range to next hour
        minutes_range = (minutes_range[0] + small_round_minutes, minutes_range[1] + small_round_minutes)

    return np.array(hourly_history_data, dtype=list), np.array(hourly_user_id_list, dtype=list), user_id_list

@funcCall
def get_statistic_data(hourly_history_data: np.array, statistic_data: np.array) -> np.array:
    '''Build/Update statistic data from hourly_history_data.'''
    if not statistic_data:
        return np.mean(hourly_history_data, axis=0)
    return statistic_data * (1 -_alpha) + hourly_history_data[-1] * _alpha

@funcCall
def createVM(machine_attributes: dict) -> dict:
    '''Create the VM instances.'''
    vm_list = {}
    machine_id_list = machine_attributes.keys()

    for id in machine_id_list:
        vm_list[id] = VM(machine_attributes[id])
    return vm_list

@funcCall
def update_user_to_vm(vm_list: dict, user_id_list: np.array) -> None:
    '''Build user to vm table.'''
    vm_id_list = vm_list.keys()

    for vm_id in vm_id_list:
        for user_id in user_id_list:
            user_to_vm = vm_list[vm_id].from_user
            if user_id not in user_to_vm:
                user_to_vm[user_id] = {
                    'bw_up':np.random.uniform(generated_bw_min, generated_bw_max),
                    'bw_down':np.random.uniform(generated_bw_min, generated_bw_max),
                    'delay':np.random.uniform(generated_delay_min, generated_delay_max)
                }

# load data & initialization
dir = './data/case1/'
machine_attributes = load_machine_data(dir + 'machine_attributes.json')
history_data = load_task_data(dir + 'history_data.json')
task_events = load_task_data(dir + 'task_events.json')
vm_list = createVM(machine_attributes)
system_time = 0
hourly_history_data, hourly_user_list, user_id_list = data_preprocessing(history_data, system_time)
mvno = MVNO()
mno = MNO(mvno, list(vm_list.keys()))

statistic_data = None
while system_time % big_round_minutes == 0:
    # prepare data
    statistic_data = get_statistic_data(hourly_history_data, statistic_data)
    update_user_to_vm(vm_list, user_id_list)

    # VM Assignment
    mno.vm_assignment(statistic_data, vm_list)

    # Task Deployment
    mno.task_deployment(task_events, vm_list)
    mvno.task_deployment(task_events, vm_list)
    break

    # update system time
    system_time += small_round_minutes