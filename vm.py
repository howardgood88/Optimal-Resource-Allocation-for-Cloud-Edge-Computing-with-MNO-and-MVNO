class VM:
    def __init__(self, attributes):
        # attributes in machine_attributes.json
        self.task_type = attributes['task_type']
        self.location = attributes['location']
        self.cpu_capacity = attributes['cpu_capacity']
        self.price = attributes['price']
        # the bw and daley to user (runtime generate in main.update_user_to_vm())
        self.to_user = {}
        # the average bw of all user (runtime generate in network_operator.MNO.vm_assignment.make_vm_bw())
        self.avg_bw = None