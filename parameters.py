from enum import (IntEnum, unique)

rnd_seed = 1126

# round time
small_round_minutes = 60
# 60 * 24 * 7
big_round_minutes = 10080
generated_bw_max = 1000
generated_bw_min = 0
generated_delay_max = 5
generated_delay_min = 0

# contract
bw_low = 0
bw_high = 10000
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
class Task_type(IntEnum):
    '''The task name maps to vector index.'''
    VoIP = 0
    IP_Video = 1
    FTP = 2