import json
import numpy as np
from network_operator import (MNO)
from vm import VM
from task_handler import Task_handler
from utils import (toSoftmax, step_logger, beta, PT5, Metrics)
from parameters import *

# initial setting for logging
import logging
if logging_level == logging.INFO:
    lev = 'INFO'
elif logging_level == logging.DEBUG:
    lev = 'DEBUG'
logging.basicConfig(format=f'%(levelname)s:Time %(system_time)s:%(message)s', filename=test_data_dir + f'log_{lev}.txt'
                    , filemode='w', level=logging_level)
## set the logger filter for showing system time
logger = logging.getLogger('root')
class ContextFileter(logging.Filter):
    def filter(self, record):
        record.system_time = Global.system_time
        return True
logger.addFilter(ContextFileter())

# fix numpy random seed and showing format
np.random.seed(rnd_seed)
np.set_printoptions(precision=2, suppress=True)

def load_task_data(filename: str) -> np.array:
    '''Load task_events.json and history_data.json from filename.'''
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data, dtype=list)

def load_machine_data(filename: str) -> dict:
    '''Load machine_attributes.json from filename.'''
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def createVM(machine_attributes: dict) -> dict:
    '''Create dict that map from vm_id to VM instance.'''
    vm_list = {}
    machine_id_list = machine_attributes.keys()
    for id in machine_id_list:
        vm_list[id] = VM(machine_attributes[id])
    logging.info('Finished creating vm instance, store in vm_list that map from vm_id to vm instance.')
    return vm_list

def get_hourly_statistic_data(hour_events: np.array) -> np.array:
    '''
    Sum up average_cpu_usage, bw_up, bw_down of hour tasks as hourly statistic data.
    For N = number of events in the hour.
    
    Parameters
    ----------
    hour_events : np.array, shape: (N, 10)
        Task-level hour data.

    Returns
    ----------
    statistic_data : np.array, shape: (3,3)
        The hourly average history data.
    '''
    task_type_idx = Task_event_index.task_type.value
    average_cpu_usage_idx = Task_event_index.average_cpu_usage.value
    T_down_idx = Task_event_index.T_down.value
    # only get the income events for traffic statistic
    hour_events = hour_events[hour_events[:, Task_event_index.event_type.value] == Event_type.start]

    hourly_statistic_data = [None for i in range(len(Task_type_index))]
    for task_type, task_idx in Task_type_index.__members__.items():
        # [average_cpu, bw_up, bw_down]
        data = hour_events[(hour_events[:, task_type_idx] == task_type)][:, average_cpu_usage_idx:T_down_idx + 1]
        data = np.zeros(3) if data.size == 0 else np.sum(data, axis=0)
        hourly_statistic_data[task_idx.value] = data
    return np.array(hourly_statistic_data, dtype=list)

def data_preprocessing(history_data: np.array) -> tuple[np.array, set]:
    '''
    Transform task-level history data into hourly history data, and generate user list.
    For N = number of tasks, H = number of hours.

    Parameters
    ----------
    history_data : np.array, shape: (N, 10)
        Task-level history data.
    system_time : int
        The discrete system time.
    
    Returns
    ----------
    hourly_history_data : np.array, shape: (H,3,3)
        The hourly average history data.
    user_list : set
        All user appeared in history data.
    '''
    # each row with hourly VoIP, IP_Video, FTP data
    hourly_history_data = []
    user_id_set = set()
    minutes_range = (Global.system_time, Global.system_time + small_round_minutes)

    start_time_idx = Task_event_index.event_time.value
    user_id_idx = Task_event_index.user_id.value
    while history_data[history_data[:, start_time_idx] > minutes_range[0]].size != 0:
        # hourly mask that fit the minutes_range
        hour_mask = (minutes_range[0] <= history_data[:, start_time_idx]) & (history_data[:, start_time_idx] < minutes_range[1])
        hour_tasks = history_data[hour_mask]
        # calculate hourly average_cpu_usage, bw_up, bw_down of different task from history data
        hourly_statistic_data = get_hourly_statistic_data(hour_tasks)
        hourly_history_data.append(hourly_statistic_data)
        # user had appeared
        user_id_set = user_id_set | set(history_data[hour_mask][:, user_id_idx])
        # update minutes_range to next hour
        minutes_range = (minutes_range[0] + small_round_minutes, minutes_range[1] + small_round_minutes)
    logging.info('Finished data preprocessing, get hour_task_record as hourly history data and record user has appeared.')
    return np.array(hourly_history_data, dtype=list), user_id_set

def generate_user_to_vm_data(location: str) -> dict:
    '''Random generate the runtime data from user to vm when new user arrive.'''
    if location == 'cloud':
        return {
                    'bw_up':beta(beta_a, beta_b, beta_t, beta_d),
                    'bw_down':beta(beta_a, beta_b, beta_t, beta_d),
                    'delay':PT5(PT5_cloud_a, PT5_cloud_b, PT5_cloud_d)
                }
    elif location == 'edge':
        return {
                    'bw_up':beta(beta_a, beta_b, beta_t, beta_d) * 0.6,
                    'bw_down':beta(beta_a, beta_b, beta_t, beta_d) * 0.6,
                    'delay':PT5(PT5_edge_a, PT5_edge_b, PT5_edge_d)
                }
    else:
        raise ValueError(f'invalid value {location} of location')

def update_user_to_vm(user_id_list: np.array) -> None:
    '''Build user to vm table.'''
    for vm_id in vm_list:
        vm = vm_list[vm_id]
        for user_id in user_id_list:
            vm.from_user.setdefault(user_id, generate_user_to_vm_data(vm.location))
    logging.info('New user/users detected, finished updating vm_list.from_user.')

def update_history_data(hourly_history_data: np.array, hour_task_record: np.array, statistic_data: np.array) -> tuple[np.array, np.array]:
    '''Append hour_task_record into hourly_history_data, and update statistic_data by hour_task_record.'''
    if hourly_history_data is None:
        # the first time call this function (cannot use vstack when hourly_history_data is empty)
        hourly_history_data = hour_task_record
        statistic_data = np.mean(hour_task_record, axis=0)
    else:
        hourly_history_data = np.vstack([hourly_history_data, hour_task_record])
    _message = f'add hour data:\n{np.mean(hour_task_record, axis=0)}\ninto statistic data:\n{statistic_data}\n'
    # update statistic data with the influence of _phi
    statistic_data = statistic_data * (1 - phi) + np.mean(hour_task_record, axis=0) * phi
    logging.info(_message + f'statistic data becomes:\n{statistic_data}')
    return hourly_history_data, statistic_data

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

def task_deployment(hour_events: np.array, minutes_range: tuple) -> None:
    '''Random assign task to operator and deploy the task.'''
    # start hourly task deployment
    global task_events
    idx = 0
    while idx < len(hour_events):
        event = hour_events[idx]
        Global.system_time = event[Task_event_index.event_time.value]
        user_id = event[Task_event_index.user_id.value]
        if user_id in bad_user:
            # bad user will not choose MNO cus it's too expansive
            idx += 1
            continue
        if event[Task_event_index.event_type.value] == Event_type.start:
            # generate the user to vm data if new user not in history data arrival
            if user_id not in user_id_set:
                if np.random.choice([True, False], 1, p = [bad_user_rate, 1 - bad_user_rate])[0]:
                    update_user_to_vm([user_id])
                    user_id_set.add(user_id)
                else:
                    bad_user.add(user_id)
                    idx += 1
                    continue
            # delegate to operator
            mno.deploy_task(event, vm_list)
        elif event[Task_event_index.event_type.value] == Event_type.end:
            mno.release_task(event)

        if Task_handler.changed:
            task_events = Task_handler.task_events
            event_time_idx = Task_event_index.event_time.value
            hour_mask = (minutes_range[0] <= task_events[:, event_time_idx]) & (task_events[:, event_time_idx] < minutes_range[1])
            hour_events = task_events[hour_mask]
            Task_handler.changed = False
        else:
            idx += 1

# load data
with step_logger('Start of load data', title1, 'Finished loading data.'):
    machine_attributes = load_machine_data(test_data_dir + 'machine_attributes.json')
    history_data = load_task_data(test_data_dir + 'history_data.json')
    task_events = load_task_data(test_data_dir + 'task_events.json')

# initialization
with step_logger('Start of Initialization', title1, 'Finished initialization.'):
    Global.system_time = 0
    # create dict map from vm_id to VM instance
    vm_list = createVM(machine_attributes)
    for vm in vm_list.values():
        vm.price = vm.price / expected_max_vm_num
    hour_task_record, user_id_set = data_preprocessing(history_data)
    # sort the set to make simulation reproducible. (or will get different user_to_vm)
    user_id_list = np.array(sorted(user_id_set))
    user_id_set = set()
    bad_user = set()
    # build user_to_vm
    update_user_to_vm(user_id_list)
    # make MNO instance
    mno = MNO(list(vm_list.keys()))
    # save the overall task events to process the unsatisfied tasks.
    Task_handler.task_events = task_events

# initialize
start_time = Global.system_time
while Global.system_time // big_round_minutes < big_round_times:
    round = Global.system_time // big_round_minutes + 1
    with step_logger(f'Start of Round {round}', title1, f'Finished Round {round}.'):
        get_avg_vm_bw()
        mno.profit = 0
        with step_logger('Start of Task Deployment', title2, f'Finished Task Deployment.'):
            hour_task_record = []
            while Global.system_time == start_time or Global.system_time % big_round_minutes != 0:
                hour_num = Global.system_time // small_round_minutes + 1
                temp_time = Global.system_time
                # get the hour tasks data
                minutes_range = (Global.system_time, Global.system_time + small_round_minutes)
                event_time_idx = Task_event_index.event_time.value
                hour_mask = (minutes_range[0] <= task_events[:, event_time_idx]) & (task_events[:, event_time_idx] < minutes_range[1])
                hour_events = task_events[hour_mask]
                
                with mno._task_deployment, step_logger(f'Start of hour {hour_num}\n'
                    f'Get hour events: {len(hour_events)}\nid,type,time\n{hour_events}', 0, f'Finished hour {hour_num}'):
                    if not hour_events.size == 0:
                        task_deployment(hour_events, minutes_range)
                        mno.profit += mno._task_deployment.user_cost
                task_events = Task_handler.task_events
                Task_handler.changed = False

                # prepare for next round
                Global.system_time = temp_time + small_round_minutes
                logging.info(f'mno overall hour utility: {mno._task_deployment.hour_utility}, # of task: {mno._task_deployment.hour_task_num}, hour fitness: {mno._task_deployment.hour_fitness}')

                hourly_statistic_data = get_hourly_statistic_data(hour_events)
                Metrics.hour_data.append(hourly_statistic_data)
        Metrics.mno_profit.append(mno.profit)
        start_time = Global.system_time
        assert(Global.system_time % big_round_minutes == 0)
logging.info(f'Finished simulating, save log to {test_data_dir}log_{lev}.txt')
print(f'Finished simulating, save log to {test_data_dir}log_{lev}.txt!')
Metrics.plot()