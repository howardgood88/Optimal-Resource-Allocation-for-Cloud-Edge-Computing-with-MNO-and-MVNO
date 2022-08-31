import abc
import numpy as np
from vm_assignment import VMAssignment
from task_deployment import TaskDeployment
from contract import Contract
from utils import (step_logger, get_total_resource, timer, Metrics)
from parameters import (_mu, title4, mno_op_bw, mno_op_cr, mvno_op_bw, mvno_op_cr, expected_task_num, Task_type_index, case_num, _theta)
import logging

class Network_operator(abc.ABC):
    def __init__(self):
        self.hold_vm_id = None
        self._task_deployment = TaskDeployment(self.name, self.op_bw, self.op_cr)

    def deploy_task(self, task: np.array, vm_list: dict) -> None:
        '''Delegate to class TaskDeployment.'''
        self._task_deployment.deploy(self.hold_vm_id, task, vm_list)

    def release_task(self, task: np.array) -> None:
        '''Delegate to class TaskDeployment.'''
        self._task_deployment.release(task)

    @timer
    def update_task_deployment_parameters(self) -> None:
        '''Delegate to class TaskDeployment.'''
        with step_logger(f'updating {self.name} parameters', title4, f'Finished updating {self.name} parameters'):
            self._task_deployment.update_parameters()

class MVNO(Network_operator):
    def __init__(self):
        self.name = 'MVNO'
        self.op_bw = mvno_op_bw
        self.op_cr = mvno_op_cr
        super().__init__()

class MNO(Network_operator):
    def __init__(self, mvno: MVNO, vm_id_list: list, vm_list: dict):
        self.name = 'MNO'
        self.op_bw = mno_op_bw
        self.op_cr = mno_op_cr
        super().__init__()
        self.mvno = mvno
        # the id of all vm own by MNO, transform to list because vm_id_list never changes.
        self.total_vm_id = np.array(vm_id_list, dtype=list)
        self.contract = Contract()
        self._vm_assignment = None

    @timer
    def vm_assignment(self, statistic_data: np.array, vm_list: dict) -> None:
        '''Calculate average vm bw and delegate to class VMAssignment.'''
        def get_avg_vm_bw():
            '''Calculate the average bw from all user to vm.'''
            for vm in vm_list.values():
                bw_up_sum = 0
                bw_down_sum = 0
                for data in vm.from_user.values():
                    bw_up_sum += data['bw_up']
                    bw_down_sum += data['bw_down']
                vm.avg_bw_up = bw_up_sum / len(vm.from_user)
                vm.avg_bw_down = bw_down_sum / len(vm.from_user)
        get_avg_vm_bw()
        self._vm_assignment = VMAssignment(self.contract, self.total_vm_id, vm_list)
        self.hold_vm_id, self.mvno.hold_vm_id = self._vm_assignment.run(statistic_data)
        # MNO
        mno_resource, edge_num, cloud_num = get_total_resource(self.hold_vm_id, vm_list)
        logging.info(f'mno vm id: {self.hold_vm_id},\ntotal resource (cr, bw_up, bw_down): {mno_resource},\n'
                        f'cloud num: {cloud_num}, edge num: {edge_num}')
        Metrics.mno_vm_resource.append(mno_resource)
        # MVNO
        mvno_resource, edge_num, cloud_num = get_total_resource(self.mvno.hold_vm_id, vm_list)
        mvno_cost = self._vm_assignment.vm_highest_price - self._vm_assignment.optimizing.best_fitness
        logging.info(f'mvno vm id: {self.mvno.hold_vm_id},\ntotal resource (cr, bw_up, bw_down): {mvno_resource},\n'
                        f'cloud num: {cloud_num}, edge num: {edge_num}, cost: {mvno_cost}')
        Metrics.mvno_vm_resource.append(mvno_resource)
        Metrics.mvno_vm_cost.append(mvno_cost)

        logging.info(f'contract: bw_high: {self.contract.bw_high}, bw_low: {self.contract.bw_low}, cr_high: {self.contract.cr_high}, cr_low: {self.contract.cr_low}')
        
        # the price mvno sells to its customers
        mvno_vm_type_num = [0, 0, 0]
        mvno_vm_cpu = [0, 0, 0]
        mvno_vm_T_up = [0, 0, 0]
        mvno_vm_T_down = [0, 0, 0]
        for id in self.mvno.hold_vm_id:
            vm = vm_list[id]
            vm.price = vm.origin_price * _mu / expected_task_num
            mvno_vm_type_num[Task_type_index[vm.task_type]] += 1
            mvno_vm_cpu[Task_type_index[vm.task_type]] += vm.cr
            mvno_vm_T_up[Task_type_index[vm.task_type]] += vm.avg_bw_up
            mvno_vm_T_down[Task_type_index[vm.task_type]] += vm.avg_bw_down
        logging.info(f'MVNO CPU demand proportion before Allocation: VoIP {statistic_data[0, 0] / np.sum(statistic_data[:, 0])}, IP Video {statistic_data[1, 0] / np.sum(statistic_data[:, 0])}, FTP {statistic_data[2, 0] / np.sum(statistic_data[:, 0])}')
        logging.info(f'MVNO CPU demand before Allocation: VoIP {statistic_data[0, 0] * _theta}, IP Video {statistic_data[1, 0] * _theta}, FTP {statistic_data[2, 0] * _theta}')
        logging.info(f'MVNO T up demand proportion before Allocation: VoIP {statistic_data[0, 1] / np.sum(statistic_data[:, 1])}, IP Video {statistic_data[1, 1] / np.sum(statistic_data[:, 1])}, FTP {statistic_data[2, 1] / np.sum(statistic_data[:, 1])}')
        logging.info(f'MVNO T up demand before Allocation: VoIP {statistic_data[0, 1] * _theta}, IP Video {statistic_data[1, 1] * _theta}, FTP {statistic_data[2, 1] * _theta}')
        logging.info(f'MVNO T down demand proportion before Allocation: VoIP {statistic_data[0, 2] / np.sum(statistic_data[:, 2])}, IP Video {statistic_data[1, 2] / np.sum(statistic_data[:, 2])}, FTP {statistic_data[2, 2] / np.sum(statistic_data[:, 2])}')
        logging.info(f'MVNO T down demand before Allocation: VoIP {statistic_data[0, 2] * _theta}, IP Video {statistic_data[1, 2] * _theta}, FTP {statistic_data[2, 2] * _theta}')
        logging.info(f'MVNO VM number proportion after Allocation: VoIP {mvno_vm_type_num[0] / np.sum(mvno_vm_type_num)}, IP Video {mvno_vm_type_num[1] / np.sum(mvno_vm_type_num)}, FTP {mvno_vm_type_num[2] / np.sum(mvno_vm_type_num)}')
        logging.info(f'MVNO VM CPU proportion after Allocation: VoIP {mvno_vm_cpu[0] / np.sum(mvno_vm_cpu)}, IP Video {mvno_vm_cpu[1] / np.sum(mvno_vm_cpu)}, FTP {mvno_vm_cpu[2] / np.sum(mvno_vm_cpu)}')
        logging.info(f'MVNO VM CPU after Allocation: VoIP {mvno_vm_cpu[0]}, IP Video {mvno_vm_cpu[1]}, FTP {mvno_vm_cpu[2]}')
        logging.info(f'MVNO VM T up proportion after Allocation: VoIP {mvno_vm_T_up[0] / np.sum(mvno_vm_T_up)}, IP Video {mvno_vm_T_up[1] / np.sum(mvno_vm_T_up)}, FTP {mvno_vm_T_up[2] / np.sum(mvno_vm_T_up)}')
        logging.info(f'MVNO VM T up after Allocation: VoIP {mvno_vm_T_up[0]}, IP Video {mvno_vm_T_up[1]}, FTP {mvno_vm_T_up[2]}')
        logging.info(f'MVNO VM T down proportion after Allocation: VoIP {mvno_vm_T_down[0] / np.sum(mvno_vm_T_down)}, IP Video {mvno_vm_T_down[1] / np.sum(mvno_vm_T_down)}, FTP {mvno_vm_T_down[2] / np.sum(mvno_vm_T_down)}')
        logging.info(f'MVNO VM T down after Allocation: VoIP {mvno_vm_T_down[0]}, IP Video {mvno_vm_T_down[1]}, FTP {mvno_vm_T_down[2]}')
        np.save(f'Metrics/{case_num}mvno_vm_type_num', mvno_vm_type_num)
        mno_vm_type_num = [0, 0, 0]
        mno_vm_cpu = [0, 0, 0]
        mno_vm_T_up = [0, 0, 0]
        mno_vm_T_down = [0, 0, 0]
        for id in self.hold_vm_id:
            vm = vm_list[id]
            vm.price = vm.origin_price / expected_task_num
            mno_vm_type_num[Task_type_index[vm.task_type]] += 1
            mno_vm_cpu[Task_type_index[vm.task_type]] += vm.cr
            mno_vm_T_up[Task_type_index[vm.task_type]] += vm.avg_bw_up
            mno_vm_T_down[Task_type_index[vm.task_type]] += vm.avg_bw_down
        logging.info(f'MNO VM proportion after Allocation: VoIP {mno_vm_type_num[0] / np.sum(mno_vm_type_num)}, IP Video {mno_vm_type_num[1] / np.sum(mno_vm_type_num)}, FTP {mno_vm_type_num[2] / np.sum(mno_vm_type_num)}')
        logging.info(f'MNO VM CPU proportion after Allocation: VoIP {mno_vm_cpu[0] / np.sum(mno_vm_cpu)}, IP Video {mno_vm_cpu[1] / np.sum(mno_vm_cpu)}, FTP {mno_vm_cpu[2] / np.sum(mno_vm_cpu)}')
        logging.info(f'MNO VM CPU after Allocation: VoIP {mno_vm_cpu[0]}, IP Video {mno_vm_cpu[1]}, FTP {mno_vm_cpu[2]}')
        logging.info(f'MNO VM T up proportion after Allocation: VoIP {mno_vm_T_up[0] / np.sum(mno_vm_T_up)}, IP Video {mno_vm_T_up[1] / np.sum(mno_vm_T_up)}, FTP {mno_vm_T_up[2] / np.sum(mno_vm_T_up)}')
        logging.info(f'MNO VM T up after Allocation: VoIP {mno_vm_T_up[0]}, IP Video {mno_vm_T_up[1]}, FTP {mno_vm_T_up[2]}')
        logging.info(f'MNO VM T down proportion after Allocation: VoIP {mno_vm_T_down[0] / np.sum(mno_vm_T_down)}, IP Video {mno_vm_T_down[1] / np.sum(mno_vm_T_down)}, FTP {mno_vm_T_down[2] / np.sum(mno_vm_T_down)}')
        logging.info(f'MNO VM T down after Allocation: VoIP {mno_vm_T_down[0]}, IP Video {mno_vm_T_down[1]}, FTP {mno_vm_T_down[2]}')
        np.save(f'Metrics/{case_num}mno_vm_type_num', mno_vm_type_num)