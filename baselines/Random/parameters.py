from enum import (IntEnum, unique)
import logging

class Global:
    system_time = 0

rnd_seed = 1127
logging_level = logging.INFO
case_num = 'case4/'
test_data_dir = './data/' + case_num
# the length of log with filling "-"
title1, title2, title3, title4, title5 = 130, 115, 100, 85, 70
# round time
small_round_minutes = 60 * 60 # s
# 60 * 24 * 7
big_round_minutes = small_round_minutes * 24
big_round_times = 7

# user to vm generating data
beta_a = 2
beta_b = 1.5
beta_t = 100
beta_d = 0
PT5_cloud_a = 2
PT5_cloud_b = 0.557
PT5_cloud_d = 49.443
PT5_edge_a = 2
PT5_edge_b = 0.557
PT5_edge_d = 1.443

# contract
expected_max_vm_num = 200
expected_min_vm_num = 5
bw_low = 100000 * expected_min_vm_num
bw_high = 100000 * expected_max_vm_num # Kbps
cr_low = 1 * expected_min_vm_num
cr_high = 1 * expected_max_vm_num

# vm assignment
mno_rate = 0.6
_theta = 0
_lambda = 0.6
_mu = 0.8

# task deployment
gamma = [
    [0.5, 1.5, 1.0, 2.0, 3.0], # VoIP
    [0.5, 3.0, 2.0, 2.0, 2.0], # IP Video
    [0.5, 3.0, 1.0, 2.0, 1.0] # FTP
]
op_bw = 300 # Kbps
op_cr = 0.05 # GCU-second/second

# utility function
max_score = 100
voip_bw_up_bmin = 370 # Kbps
voip_bw_down_bmin = 16.5
ipVideo_bw_up_bmax = 20
ipVideo_bw_down_bmax = 950
ftp_bw_up_bmin = 5
ftp_bw_up_bmax = 35
ftp_bw_down_bmin = 100
ftp_bw_down_bmax = 1200

# update history data
phi = 0.9

# optimizing
## the max time of searching the valid offsprings in vm assignment
max_searching_times = 1000
optimizing_times = 100
offspring_number = 5
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
    cpu_request = 5
    average_cpu_usage = 6
    T_up = 7
    T_down = 8