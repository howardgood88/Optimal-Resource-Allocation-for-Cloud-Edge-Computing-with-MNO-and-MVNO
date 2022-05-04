import json
import numpy as np
from utils import (printReturn, funcCall)
from network_operator import (MNO, MVNO)
from parameters import (small_round_minutes, big_round_minutes, rnd_seed, Task_type_index, Task_event_index, _alpha,
                        generated_bw_max, generated_bw_min, generated_delay_max, generated_delay_min,
                        big_round_number)
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

def get_hourly_statistic_data(hourly_tasks: np.array) -> np.array:
    '''Calculate hourly average_cpu_usage, bw_up, bw_down of different task from history data'''
    task_type_idx = Task_event_index.task_type.value
    average_cpu_usage_idx = Task_event_index.average_cpu_usage.value
    T_down_idx = Task_event_index.T_down.value

    get_hourly_statistic_data = [None for i in range(len(Task_type_index))]
    for task_type, task_idx in Task_type_index.__members__.items():
        # [average_cpu, bw_up, bw_down]
        data = hourly_tasks[(hourly_tasks[:, task_type_idx] == task_type)][:, average_cpu_usage_idx:T_down_idx + 1]
        data = np.zeros(3) if data.size == 0 else np.mean(data, axis=0)
        get_hourly_statistic_data[task_idx.value] = data
    return get_hourly_statistic_data

@funcCall
def data_preprocessing(history_data: np.array, system_time: int) -> tuple[np.array, np.array, np.array]:
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
    # index of task_event.json/history_data.json
    start_time_idx = Task_event_index.start_time.value
    user_id_idx = Task_event_index.user_id.value
    while history_data[history_data[:, start_time_idx] > minutes_range[0]].size != 0:
        # hourly mask that fit the minutes_range
        hour_mask = (minutes_range[0] <= history_data[:, start_time_idx]) & (history_data[:, start_time_idx] < minutes_range[1])
        hour_tasks = history_data[hour_mask]

        # calculate hourly average_cpu_usage, bw_up, bw_down of different task from history data
        hourly_statistic_data = get_hourly_statistic_data(hour_tasks)
        hourly_history_data.append(hourly_statistic_data)

        # user appeared in this hour
        users = set(history_data[hour_mask][:, user_id_idx])
        hourly_user_id_list.append(users)

        # user had appeared
        user_id_list = user_id_list | users

        # update minutes_range to next hour
        minutes_range = (minutes_range[0] + small_round_minutes, minutes_range[1] + small_round_minutes)

    return np.array(hourly_history_data, dtype=list), np.array(hourly_user_id_list, dtype=list), np.array(sorted(user_id_list))

@funcCall
def update_data(hourly_history_data: np.array, hour_task_record: np.array, statistic_data: np.array) -> tuple[np.array, np.array]:
    '''Build/Update statistic data from hourly_history_data.'''
    if hourly_history_data is None:
        hourly_history_data = hour_task_record
    else:
        hourly_history_data = np.vstack([hourly_history_data, hour_task_record])
    return hourly_history_data, statistic_data * (1 -_alpha) + np.mean(hour_task_record, axis=0) * _alpha

@funcCall
def createVM(machine_attributes: dict) -> dict:
    '''Create the VM instances.'''
    vm_list = {}
    machine_id_list = machine_attributes.keys()

    for id in machine_id_list:
        vm_list[id] = VM(machine_attributes[id])
    return vm_list

def generate_user_to_vm_data():
    return {
                'bw_up':np.random.uniform(generated_bw_min, generated_bw_max),
                'bw_down':np.random.uniform(generated_bw_min, generated_bw_max),
                'delay':np.random.uniform(generated_delay_min, generated_delay_max)
            }

@funcCall
def update_user_to_vm(vm_list: dict, user_id_list: np.array) -> None:
    '''Build user to vm table.'''
    vm_id_list = vm_list.keys()

    for vm_id in vm_id_list:
        for user_id in user_id_list:
            user_to_vm = vm_list[vm_id].from_user
            user_to_vm.setdefault(user_id, generate_user_to_vm_data())

@funcCall
def task_deployment(hour_tasks, vm_list):
    '''Random assign task to operator and deploy the task.'''
    # try redeploy the unaccepted tasks.
    for operator in (mno, mvno):
        queue_size = operator._task_deployment.unaccepted_task_queue.qsize()
        while queue_size > 0:
            task = operator._task_deployment.unaccepted_task_queue.get()
            duration = task[Task_event_index.end_time.value] - task[Task_event_index.start_time.value]
            task[Task_event_index.start_time.value] = system_time
            task[Task_event_index.end_time.value] = system_time + duration
            operator.task_deployment(task, vm_list)
            queue_size -= 1
    for task in hour_tasks:
        operator = user_id_to_operator.setdefault(task[Task_event_index.user_id.value], np.random.choice([mno, mvno], 1, p = [0.7, 0.3])[0])
        print(f'task{task[Task_event_index.index.value]} assign to {operator.name}')
        operator.task_deployment(task, vm_list)

# load data & initialization
print('-----------Start of load data & initialization------------')
dir = './data/case1/'
machine_attributes = load_machine_data(dir + 'machine_attributes.json')
history_data = load_task_data(dir + 'history_data.json')
task_events = load_task_data(dir + 'task_events.json')
vm_list = createVM(machine_attributes)
system_time = 0
hour_task_record, hourly_user_list, user_id_list = data_preprocessing(history_data, system_time)
mvno = MVNO()
mno = MNO(mvno, list(vm_list.keys()))
user_id_to_operator = {}

statistic_data = np.zeros((3,3))
hourly_history_data = None
while system_time <= (big_round_minutes * big_round_number):
    # prepare data
    print('-----------Start of prepare data------------')
    hourly_history_data, statistic_data = update_data(hourly_history_data, hour_task_record, statistic_data)
    update_user_to_vm(vm_list, user_id_list)

    # VM Assignment
    print('-----------Start of VM Assignment------------')
    mno.vm_assignment(statistic_data, vm_list)

    # Task Deployment
    print(f'-----------Start of Task Deployment------------')
    hour_task_record = []
    while system_time == 0 or system_time % big_round_minutes != 0:
        minutes_range = (system_time, system_time + small_round_minutes)
        start_time_idx = Task_event_index.start_time.value
        hour_mask = (minutes_range[0] <= task_events[:, start_time_idx]) & (task_events[:, start_time_idx] < minutes_range[1])
        hour_tasks = task_events[hour_mask]
        if hour_tasks.size == 0:
            break
        print(f'hour tasks: {hour_tasks[:, Task_event_index.index.value]}')
                
        task_deployment(hour_tasks, vm_list)

        # update system time
        system_time += small_round_minutes

        hour_task_record.append(get_hourly_statistic_data(hour_tasks))
        print(f'-----------Start of next round, system time: {system_time}, hour: {system_time // small_round_minutes}------------')
    hourly_history_data = np.vstack([hourly_history_data, hour_task_record])
    if hour_tasks.size == 0:
        break
    assert(system_time % big_round_minutes == 0)