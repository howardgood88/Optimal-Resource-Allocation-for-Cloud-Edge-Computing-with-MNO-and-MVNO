from enum import (IntEnum, unique)

rnd_seed = 1126

# round time
small_round_minutes = 60
# 60 * 24 * 7
big_round_minutes = 10080

# contract
bw_low = 0
bw_high = 10000
cr_low = 0
cr_high = 5


# special symbol
_theta = 0.3
_lambda = 0.6
_mu = 0.8
_alpha = 0.7

@unique
class Task_type(IntEnum):
    '''The task name maps to vector index.'''
    VoIP = 0
    IP_Video = 1
    FTP = 2