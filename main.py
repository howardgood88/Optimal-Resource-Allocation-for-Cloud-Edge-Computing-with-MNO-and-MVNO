import json
import numpy as np
from network_operator import (MNO, MVNO)
from vm import VM
from task_handler import Task_handler
from utils import (toSoftmax)
from parameters import (test_data_dir, small_round_minutes, big_round_minutes, big_round_times, rnd_seed, Task_type_index, Task_event_index,
                        Event_type, _phi, generated_bw_max, generated_bw_min, generated_delay_cloud_max, generated_delay_cloud_min,
                        generated_delay_edge_max, generated_delay_edge_min, logging_level, mno_rate, Global,
                        title1, title2)
# initial setting for logging
import logging
logging.basicConfig(format=f'%(levelname)s:Time %(system_time)s:%(message)s', filename=test_data_dir + 'log.txt'
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
    return vm_list

def get_hourly_statistic_data(hour_events: np.array) -> np.array:
    '''
    Calculate hourly average_cpu_usage, bw_up, bw_down of hour tasks.
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

    hourly_statistic_data = [None for i in range(len(Task_type_index))]
    for task_type, task_idx in Task_type_index.__members__.items():
        # [average_cpu, bw_up, bw_down]
        data = hour_events[(hour_events[:, task_type_idx] == task_type)][:, average_cpu_usage_idx:T_down_idx + 1]
        data = np.zeros(3) if data.size == 0 else np.mean(data, axis=0)
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
    # all user had appeared
    user_id_set = set()
    # the selected time range
    minutes_range = (Global.system_time, Global.system_time + small_round_minutes)
    # index of task_event.json/history_data.json
    start_time_idx = Task_event_index.event_time.value
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

def generate_user_to_vm_data(location: str) -> dict:
    '''Random generate the runtime data from user to vm when new user arrive.'''
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

def update_history_data(hourly_history_data: np.array, hour_task_record: np.array, statistic_data: np.array) -> tuple[np.array, np.array]:
    '''Append hour_task_record into hourly_history_data, and update statistic_data by hour_task_record.'''
    if hourly_history_data is None:
        # the first time call this function (cannot use vstack when hourly_history_data is empty)
        hourly_history_data = hour_task_record
    else:
        hourly_history_data = np.vstack([hourly_history_data, hour_task_record])
    _message = f'add hour data\n{np.mean(hour_task_record, axis=0)} into statistic data, '
    # update statistic data with the influence of _phi
    statistic_data = statistic_data * (1 -_phi) + np.mean(hour_task_record, axis=0) * _phi
    logging.debug(_message + f'statistic data becomes:\n{statistic_data}')
    return hourly_history_data, statistic_data

def task_deployment(hour_events: np.array) -> None:
    '''Random assign task to operator and deploy the task.'''
    # initialization
    for operator in (mno, mvno):
        operator._task_deployment.hour_utility = 0
        operator._task_deployment.hour_task_num = 0
    # start hourly task deployment
    logging.info(f'mno deploy with best population {toSoftmax(mno._task_deployment.optimizing.best_population)} '
                    f'with fitness: {mno._task_deployment.optimizing.best_fitness}')
    logging.info(f'mvno deploy with best population {toSoftmax(mvno._task_deployment.optimizing.best_population)} '
                    f'with fitness: {mvno._task_deployment.optimizing.best_fitness}')
    for event in hour_events:
        Global.system_time = event[Task_event_index.event_time.value]
        user_id = event[Task_event_index.user_id.value]
        if event[Task_event_index.event_type.value] == Event_type.start:
            # get/assign the operator to the task
            operator = user_id_to_operator.setdefault(user_id, np.random.choice([mno, mvno], 1, p = [mno_rate, 1 - mno_rate])[0])
            logging.info(f'task{event[Task_event_index.index.value]} assign to {operator.name}')
            # generate the user to vm data if new user not in history data arrival
            if user_id not in user_id_set:
                update_user_to_vm([user_id])
                user_id_set.add(user_id)
            # delegate to operator
            operator.deploy_task(event, vm_list)
        elif event[Task_event_index.event_type.value] == Event_type.end:
            operator = user_id_to_operator[user_id]
            operator.release_task(event)

# load data & initialization
logging.info(f'{"Start of load data & initialization":-^{title1}}')
machine_attributes = load_machine_data(test_data_dir + 'machine_attributes.json')
history_data = load_task_data(test_data_dir + 'history_data.json')
task_events = load_task_data(test_data_dir + 'task_events.json')
logging.info('Finished loading data.')
Global.system_time = 0
# create dict map from vm_id to VM instance
vm_list = createVM(machine_attributes)
logging.info('Finished creating vm instance, store in vm_list that map from vm_id to vm instance.')
hour_task_record, user_id_set = data_preprocessing(history_data)
logging.info('Finished data preprocessing, get hour_task_record as hourly history data and record user has appeared.')
# sort the set to make simulation reproducible. (or will get different user_to_vm)
user_id_list = np.array(sorted(user_id_set))

# build user_to_vm
update_user_to_vm(user_id_list)
logging.info('Finished map from user to vm in vm_list.from_user.')
# make MNO and MVNO instance
mvno = MVNO()
mno = MNO(mvno, list(vm_list.keys()), vm_list)
# for keeping the mapping from user_id to the operator within task deployment
user_id_to_operator = {}
# save the overall task events to process the unsatisfied tasks.
Task_handler.task_events = task_events

# initialize
statistic_data = np.zeros((3,3))
hourly_history_data = None
start_time = Global.system_time
while Global.system_time // big_round_minutes < big_round_times:
    logging.info(f'{f"Start of Round {Global.system_time // big_round_minutes + 1}":-^{title1}}')
    # prepare data
    logging.info(f'{"Start of updating history data":-^{title1}}')
    # update history data and statistic data
    hourly_history_data, statistic_data = update_history_data(hourly_history_data, hour_task_record, statistic_data)
    logging.info('Finished update history data and statistic data.')

    # VM Assignment
    logging.info(f'{"Start of VM Assignment":-^{title1}}')
    mno.vm_assignment(statistic_data, vm_list)
    logging.info('Finished vm assignment.')

    # Task Deployment
    logging.info(f'{f"Start of Task Deployment round {Global.system_time // big_round_minutes + 1}":-^{title1}}')
    hour_task_record = []
    while Global.system_time == start_time or Global.system_time % big_round_minutes != 0:
        temp_time = Global.system_time
        # hourly task deployment
        logging.info(f'{f"Start of hour {Global.system_time // small_round_minutes + 1}, system time: {Global.system_time}":-^{title2}}')
        ## get the hour tasks data
        minutes_range = (Global.system_time, Global.system_time + small_round_minutes)
        event_time_idx = Task_event_index.event_time.value
        hour_mask = (minutes_range[0] <= task_events[:, event_time_idx]) & (task_events[:, event_time_idx] < minutes_range[1])
        hour_events = task_events[hour_mask]
        logging.info(f'Get hour events:\nid,type,time\n{hour_events[:, Task_event_index.index.value : Task_event_index.event_time.value + 1]}')
        if not hour_events.size == 0:
            task_deployment(hour_events)

        # prepare for next round
        Global.system_time = temp_time + small_round_minutes

        # notify operator task deployment is end and calculate the statistic performance of this hour
        mno.end_task_deployment()
        mvno.end_task_deployment()
        logging.info(f'mno overall hour utility: {mno._task_deployment.hour_utility}, hour fitness: {mno._task_deployment.hour_fitness}')
        logging.info(f'mvno overall hour utility: {mvno._task_deployment.hour_utility}, hour fitness: {mvno._task_deployment.hour_fitness}')
        logging.info(f'{"Start of Updating Parameters":-^{title1}}')
        mno.update_task_deployment_parameters()
        mvno.update_task_deployment_parameters()
        logging.info('Finished updating MNO and MVNO parameters.')

        hourly_statistic_data = get_hourly_statistic_data(hour_events)
        hour_task_record.append(hourly_statistic_data)
        logging.info('Finished task deployment.')

    hour_task_record = np.array(hour_task_record, dtype=list)
    start_time = Global.system_time
    assert(Global.system_time % big_round_minutes == 0)

logging.info(f'Finish simulating...')