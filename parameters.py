from enum import (IntEnum, unique)
import logging

class Global:
    system_time = 0

rnd_seed = 1126
logging_level = logging.DEBUG
test_data_dir = './data/case3/'
# the length of log with filling "-"
title1, title2, title3, title4, title5 = 130, 115, 100, 85, 70
# round time
small_round_minutes = 60
# 60 * 24 * 7
big_round_minutes = small_round_minutes * 2
big_round_times = 3

# user to vm generating data
generated_bw_max = 100000 # Kbps
generated_bw_min = 50000 # Kbps
generated_delay_cloud_max = 50
generated_delay_cloud_min = 49
generated_delay_edge_max = 2
generated_delay_edge_min = 1

# contract
bw_low = 5
bw_high = generated_bw_max * 10 # Kbps
cr_low = 0
cr_high = 10

# vm assignment
mno_rate = 0.7
_theta = 0.3
_lambda = 0.6
_mu = 0.8

# task deployment
_gamma = [
    [3.0, 1.0, 1.0, 3.0, 5.0, 3.0], # VoIP
    [0.5, 3.0, 2.0, 3.0, 2.0, 3.0], # IP Video
    [3.0, 0.5, 1.0, 3.0, 1.0, 3.0] # FTP
]
_op_bw = 300 # Kbps
_op_cr = 0.2 # GCU-second/second

# update history data
_phi = 0.7

# optimizing
## the max time of searching the valid offsprings in vm assignment
max_searching_times = 50
optimizing_times = 10
offspring_number = 4
mutate_rate = 0.05

@unique
class Task_type_index(IntEnum):
    '''The task name maps to vector index.'''
    VoIP = 0
    IP_Video = 1
    FTP = 2

@unique
class Event_type(IntEnum):
    '''The event type of task.'''
    start = 0
    end = 1

@unique
class Task_event_index(IntEnum):
    '''
    The attribute name maps to vector index.
    (don't use dict for keeping np.array property.)
    '''
    index = 0
    event_type = 1
    event_time = 2
    task_type = 3
    user_id = 4
    machine_id = 5
    cpu_request = 6
    average_cpu_usage = 7
    T_up = 8
    T_down = 9