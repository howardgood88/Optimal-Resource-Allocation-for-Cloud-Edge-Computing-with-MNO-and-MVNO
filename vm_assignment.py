import numpy as np
from network_operator import Contract
from parameters import Parameters

np.random.seed(Parameters.rnd_seed)

class VMAssignment:
    def __init__(self):
        self._lambda = 0.6
        self._mu = 0.8
    
    def run(self, contract: Contract, total_machine_id: np.array, statistic_data: np.array, machine_bw: dict, machine_attributes: dict):
        '''
        Start running VMAssignment algorithm.

        Parameters
        ----------
        contract : Contract
            Contract object that supply the upper bound and lower bound of bw and cr.
        total_machine_id : np.array
            The machines that MNO can supply.
        statistic_data : np.array
            The statistic data of history data.
        machine_bw : dict
            The average bw to machine of all user in user_list.
        machine_attributes : dict
            The machine attributes in machine_attributes.json.
        '''
        while 1:
            choose_machine_id = self.choose_machine(total_machine_id)
            if self.check_condition(choose_machine_id):
                break

    def choose_machine(self, total_machine_id: np.array):
        return total_machine_id[np.random.choice([True, False], total_machine_id.shape, p = [0.3, 0.7])]

    def check_condition(self, choose_machine_id: np.array):
        return True