import json
import numpy as np
from utils import (printReturn, funcCall, print_vm_list)
from network_operator import (MNO, MVNO)
from parameters import (test_data_dir, small_round_minutes, big_round_minutes, big_round_times, rnd_seed, Task_type_index, Task_event_index,
                        _phi, generated_bw_max, generated_bw_min, generated_delay_cloud_max, generated_delay_cloud_min,
                        generated_delay_edge_max, generated_delay_edge_min, logging_level)
from vm import VM
import logging

logging.basicConfig(filename = test_data_dir + 'log.txt', filemode='w', level=logging_level)
np.random.seed(rnd_seed)
np.set_printoptions(precision=2, suppress=True)

def load_task_data(filename: str) -> np.array:
    '''Load task_events.json and history_data.json from filename'''
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data, dtype=list)

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

    hourly_statistic_data = [None for i in range(len(Task_type_index))]
    for task_type, task_idx in Task_type_index.__members__.items():
        # [average_cpu, bw_up, bw_down]
        data = hourly_tasks[(hourly_tasks[:, task_type_idx] == task_type)][:, average_cpu_usage_idx:T_down_idx + 1]
        data = np.zeros(3) if data.size == 0 else np.mean(data, axis=0)
        hourly_statistic_data[task_idx.value] = data
    return np.array(hourly_statistic_data, dtype=list)

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
    # all user had appeared
    user_id_set = set()
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
        # user had appeared
        user_id_set = user_id_set | users

        # update minutes_range to next hour
        minutes_range = (minutes_range[0] + small_round_minutes, minutes_range[1] + small_round_minutes)
    return np.array(hourly_history_data, dtype=list), user_id_set

def update_data(hourly_history_data: np.array, hour_task_record: np.array, statistic_data: np.array) -> tuple[np.array, np.array]:
    '''Update hourly_history_data, and update statistic_data by hour_task_record.'''
    logging.debug(f'add hour_task_record {hour_task_record} into hourly_history_data and update statistic_data')
    if hourly_history_data is None:
        hourly_history_data = hour_task_record
    else:
        hourly_history_data = np.vstack([hourly_history_data, hour_task_record])
    logging.info(f'add hour data\n{np.mean(hour_task_record, axis=0)} into statistic data')
    statistic_data = statistic_data * (1 -_phi) + np.mean(hour_task_record, axis=0) * _phi
    logging.info(f'statistic data becomes:\n{statistic_data}')
    return hourly_history_data, statistic_data

def createVM(machine_attributes: dict) -> dict:
    '''Create dict map from vm_id to VM instance.'''
    vm_list = {}
    machine_id_list = machine_attributes.keys()

    for id in machine_id_list:
        vm_list[id] = VM(machine_attributes[id])
    return vm_list

def generate_user_to_vm_data(location: str) -> dict:
    '''Random generate the data from user to vm when new user arrival.'''
    if location == 'cloud':
        return {
                    'bw_up':np.random.uniform(generated_bw_min, generated_bw_max),
                    'bw_down':np.random.uniform(generated_bw_min, generated_bw_max),
                    'delay':np.random.uniform(generated_delay_cloud_min, generated_delay_cloud_max)
                }
    elif location == 'edge':
        return {
                    'bw_up':np.random.uniform(generated_bw_min, generated_bw_max),
                    'bw_down':np.random.uniform(generated_bw_min, generated_bw_max),
                    'delay':np.random.uniform(generated_delay_edge_min, generated_delay_edge_max)
                }
    else:
        raise ValueError(f'invalid value {location} of location')

def update_user_to_vm(user_id_list: np.array) -> None:
    '''Build user to vm table.'''
    for vm_id in vm_list:
        vm = vm_list[vm_id]
        for user_id in user_id_list:
            vm.from_user.setdefault(user_id, generate_user_to_vm_data(vm.location))

def task_deployment(hour_tasks: np.array) -> None:
    '''Random assign task to operator and deploy the task.'''
    # try to redeploy the unaccepted tasks
    for operator in (mno, mvno):
        logging.info(f'------Start trying to redeploy undone tasks of {operator.name}------')
        operator.redeploy(vm_list, system_time)
    # start hourly task deployment
    logging.info(f'mno best population {mno._task_deployment.optimizing.best_population} with fitness: {mno._task_deployment.optimizing.best_fitness}')
    logging.info(f'mvno best population {mvno._task_deployment.optimizing.best_population} with fitness: {mvno._task_deployment.optimizing.best_fitness}')
    for task in hour_tasks:
        user_id = task[Task_event_index.user_id.value]
        # get/assign the operator to the task
        operator = user_id_to_operator.setdefault(user_id, np.random.choice([mno, mvno], 1, p = [0.7, 0.3])[0])
        logging.info(f'task{task[Task_event_index.index.value]} assign to {operator.name}')
        # generate the user to vm data if new user not in history data arrival
        if user_id not in user_id_set:
            update_user_to_vm([user_id])
            user_id_set.add(user_id)
        # delegate to operator
        operator.task_deployment(task, vm_list)
    logging.info(f'mno overall utility: {mno._task_deployment.optimizing.best_fitness}')
    logging.info(f'mvno overall utility: {mvno._task_deployment.optimizing.best_fitness}')

    # update best population based on the operating(deploy utility) of this hour
    logging.info(f'------------Start of Updating best population------------')
    mno.update_task_deployment_best_population()
    mvno.update_task_deployment_best_population()

    logging.info(f'------------Start of Updating Parameters------------')
    mno.update_task_deployment_parameters()
    mvno.update_task_deployment_parameters()

# load data & initialization
logging.info('------------Start of load data & initialization------------')
machine_attributes = load_machine_data(test_data_dir + 'machine_attributes.json')
history_data = load_task_data(test_data_dir + 'history_data.json')
task_events = load_task_data(test_data_dir + 'task_events.json')
logging.info('Finished loading data.')
# create dict map from vm_id to VM instance
vm_list = createVM(machine_attributes)
logging.info('Finished creating vm instance, store in vm_list that map from vm_id to vm instance.')
system_time = 0
hour_task_record, user_id_set = data_preprocessing(history_data, system_time)
logging.info('Finished data preprocessing, get hour_task_record as hourly history data and user_id_set.')
# sort the set to make simulation reproducible. (or will get different user_to_vm)
user_id_list = np.array(sorted(user_id_set))

# build user_to_vm
update_user_to_vm(user_id_list)
logging.info('Finished building user to vm data into vm_list.from_user.')
# make MNO and MVNO instance
mvno = MVNO()
mno = MNO(mvno, list(vm_list.keys()), vm_list)
# for keeping the mapping from user_id to the operator within task deployment
user_id_to_operator = {}

# initialize
statistic_data = np.zeros((3,3))
hourly_history_data = None
start_time = system_time
while system_time // big_round_minutes < big_round_times:
    logging.info(f'-----------------Start of Round {system_time // big_round_minutes + 1}-----------------')
    # prepare data
    logging.info('------------Start of updating data------------')
    # update history data and statistic data
    hourly_history_data, statistic_data = update_data(hourly_history_data, hour_task_record, statistic_data)
    logging.info('Finished update hourly_history_data and statistic_data.')

    # VM Assignment
    logging.info('------------Start of VM Assignment------------')
    mno.vm_assignment(statistic_data, vm_list)
    logging.info('Finished vm assignment.')

    # Task Deployment
    logging.info(f'------------Start of Task Deployment round {system_time // big_round_minutes + 1}------------')
    hour_task_record = []
    while system_time == start_time or system_time % big_round_minutes != 0:
        # hourly task deployment
        logging.info(f'-------Start of hour {system_time // small_round_minutes + 1}, system time: {system_time}-------')
        ## get the hour tasks data
        minutes_range = (system_time, system_time + small_round_minutes)
        start_time_idx = Task_event_index.start_time.value
        hour_mask = (minutes_range[0] <= task_events[:, start_time_idx]) & (task_events[:, start_time_idx] < minutes_range[1])
        hour_tasks = task_events[hour_mask]
        logging.info(f'hour tasks: {hour_tasks[:, Task_event_index.index.value]}')
        if not hour_tasks.size == 0:
            task_deployment(hour_tasks)
        # prepare for next round
        system_time += small_round_minutes
        hourly_statistic_data = get_hourly_statistic_data(hour_tasks)
        hour_task_record.append(hourly_statistic_data)
    logging.info(f'Finished task deployment {system_time // big_round_minutes}.')

    hour_task_record = np.array(hour_task_record, dtype=list)
    start_time = system_time
    assert(system_time % big_round_minutes == 0)

logging.info(f'Finish simulating...')