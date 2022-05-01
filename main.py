import json
import numpy as np
from utility import (printReturn, funcCall)
from network_operator import (MNO, MVNO)
from parameters import Parameters

np.random.seed(Parameters.rnd_seed)

@funcCall
def loadTaskData(filename: str) -> np.array:
    '''Load task_events.json and history_data.json from filename'''
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data, dtype=list)

@funcCall
def loadMachineData(filename: str) -> dict:
    '''Load machine_attributes.json from filename'''
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

@funcCall
def prepareData(history_data: np.array, system_time: int) -> tuple[np.array, np.array, np.array]:
    '''Transform task-level history data into hourly history data and average statistic data.'''
    # each row with hourly VoIP, IP_Video, FTP data
    hourly_history_data = []
    # each row with the set of user appeared in the hour
    hourly_user_list = []
    # all user had appeared
    user_list = set()
    # the selected time range
    minutes_range = (system_time, system_time + 59)
    while history_data[history_data[:, 1] > minutes_range[0]].size != 0:
        minutes_mask = (minutes_range[0] <= history_data[:, 1]) & (history_data[:, 1] <= minutes_range[1])
        hourly_tasks_data = []
        for task_type in ('VoIP', 'IP_Video', 'FTP'):
            data = history_data[minutes_mask & (history_data[:, 3] == task_type)][:, -3:]
            hourly_data = np.zeros(3) if data.size == 0 else np.mean(data, axis=0)
            hourly_tasks_data.append(hourly_data)

        hourly_history_data.append(hourly_tasks_data)
        users = set(history_data[minutes_mask][:, 4])
        hourly_user_list.append(users)
        user_list = user_list | users
        minutes_range = (minutes_range[0] + 60, minutes_range[1] + 60)

    hourly_history_data = np.array(hourly_history_data, dtype=list)
    hourly_user_list = np.array(hourly_user_list, dtype=list)
    return hourly_history_data, hourly_user_list, np.mean(hourly_history_data, axis=0), user_list

@printReturn
def generateUserToMachine(machine_attributes: dict, user_list: np.array) -> dict:
    '''Build userToMachine.'''
    user_to_machine = {}
    machine_id_list = machine_attributes.keys()
    
    for user_id in user_list:
        if user_id not in user_to_machine:
            user_to_machine[user_id] = {}
        for machine_id in machine_id_list:
            user_to_machine[user_id][machine_id] = {'bw':np.random.uniform(0, 101), 'delay':np.random.uniform(0, 5)}
    return user_to_machine

dir = './data/case1/'
machine_attributes = loadMachineData(dir + 'machine_attributes.json')
history_data = loadTaskData(dir + 'history_data.json')
task_events = loadTaskData(dir + 'task_events.json')

system_time = 0
hourly_history_data, hourly_user_list, statistic_data, user_list = prepareData(history_data, system_time)
user_to_machine = generateUserToMachine(machine_attributes, user_list)

mvno = MVNO()
mno = MNO(mvno, machine_attributes.keys())
mno.vm_assignment(statistic_data, user_to_machine, machine_attributes)