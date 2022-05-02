from vm_assignment import VMAssignment
import numpy as np
from parameters import (_lambda, _mu, bw_low, bw_high, cr_low, cr_high)
from utility import (printReturn, funcCall)

class Contract:
    def __init__(self):
        self.bw_low = bw_low
        self.bw_high = bw_high
        self.cr_low = cr_low
        self.cr_high = cr_high

class MVNO:
    def __init__(self):
        self.hold_vm_id = None

class MNO:
    def __init__(self, mvno: MVNO, vm_id_list: list):
        self.mvno = mvno
        self._vm_assignment = VMAssignment()
        # the id of all vm own by MNO, transform to list because vm_id_list never changes.
        self.total_vm_id = np.array(vm_id_list, dtype=list)
        self.hold_vm_id = None

    @funcCall
    def vm_assignment(self, statistic_data: np.array, vm_list: dict) -> None:
        '''Process the data to needed format and delegate to class VMAssignment.'''
        def get_avg_vm_bw(vm_list: dict):
            '''Calculate the average bw from all reachable user to vm.'''
            for vm in vm_list.values():
                bw_sum = 0
                for data in vm.to_user.values():
                    bw_sum += data['bw']
                vm.avg_bw = bw_sum / len(vm.to_user)

        self.contract = Contract()
        get_avg_vm_bw(vm_list)
        self.hold_vm_id, self.mvno.hold_vm_id = self._vm_assignment.run(self.contract, self.total_vm_id,
                                                            statistic_data, vm_list)
        print(f'mno vm id: {self.hold_vm_id}, mvno vm id: {self.mvno.hold_vm_id}')
        
        # the price mvno sells to its customers
        for id in self.mvno.hold_vm_id:
            vm_list[id].price = vm_list[id].price * _mu