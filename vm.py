class VM:
    '''The attributes of vm.'''
    def __init__(self, attributes):
        # attributes in machine_attributes.json
        self.id = attributes['id']
        self.task_type = attributes['task_type']
        self.location = attributes['location']
        self.cr = attributes['cpu_capacity']
        self.price = attributes['price']
        # the bw and daley to user (runtime generate in main.update_user_to_vm())
        self.from_user = {}
        # the average bw of all user (runtime generate in network_operator.MNO.vm_assignment.make_vm_bw())
        self.avg_bw_up = None
        self.avg_bw_down = None