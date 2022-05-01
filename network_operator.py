from vm_assignment import VMAssignment
import numpy as np

class Contract:
    def __init__(self):
        self.bw_low = 0
        self.bw_high = 100
        self.cr_low = 0
        self.cr_high = 1

class MVNO:
    def __init__(self):
        pass

class MNO:
    def __init__(self, mvno: MVNO, machine_id_list: dict):
        self.mvno = mvno
        self.vm_assignment = VMAssignment()
        # the id of all vm own by MNO, transform to list because machine_list never changes.
        self.total_machine_id = np.array(machine_id_list)
    
    def vm_assignment(self, contract: Contract, statistic_data: np.array, user_to_machine: dict, machine_attributes: dict):
        '''Process the data to needed format and delegate to class VMAssignment.'''
        def make_machine_bw(user_to_machine: dict):
            machine_bws = {}
            for to_machine in user_to_machine.values():
                for machine_id, data in to_machine.items():
                    if machine_id not in machine_bws:
                        machine_bws[machine_id] = []
                    machine_bws[machine_id].append(data['bw'])

            machine_bw = {}
            for machine_id, bws in machine_bws.items():
                machine_bw[machine_id] = sum(bws) / len(bws)

            return machine_bw

        self.contract = Contract()
        machine_bw = make_machine_bw(user_to_machine)
        self.vm_assignment.run(contract, self.total_machine_id, statistic_data, machine_bw, machine_attributes)
