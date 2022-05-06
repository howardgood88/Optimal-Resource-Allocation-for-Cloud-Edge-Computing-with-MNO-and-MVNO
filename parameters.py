from enum import (IntEnum, unique)

rnd_seed = 1126

test_data_dir = './data/case1/'
# round time
small_round_minutes = 60
# 60 * 24 * 7
big_round_minutes = small_round_minutes * 2
generated_bw_max = 100000 # Kbps
generated_bw_min = 0 # Kbps
generated_delay_cloud_max = 5
generated_delay_cloud_min = 0
generated_delay_edge_max = 5
generated_delay_edge_min = 0

# contract
bw_low = 0
bw_high = generated_bw_max * 100 # Kbps
cr_low = 0
cr_high = 5

# vm assignment
_theta = 0.3
_lambda = 0.6
_mu = 0.8

# task deployment
_gamma = [
    [3, 1, 1, 3, 5, 3], # VoIP
    [0.5, 3, 2, 3, 2, 3], # IP Video
    [3, 0.5, 1, 3, 1, 3] # FTP
]
_op_bw = 300 # Kbps
_op_cr = 0.2 # GCU-second/second

# update history data
_alpha = 0.7

@unique
class Task_type_index(IntEnum):
    '''The task name maps to vector index.'''
    VoIP = 0
    IP_Video = 1
    FTP = 2

@unique
class Task_event_index(IntEnum):
    '''The attribute name maps to vector index. (don't use dict for keeping np.array property.)'''
    index = 0
    round = 1
    start_time = 2
    end_time = 3
    task_type = 4
    user_id = 5
    machine_id = 6
    cpu_request = 7
    average_cpu_usage = 8
    T_up = 9
    T_down = 10
    operator = 11 # runtime
    running_at = 12 # runtime