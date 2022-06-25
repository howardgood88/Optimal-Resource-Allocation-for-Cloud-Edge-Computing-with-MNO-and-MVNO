class VM:
    '''The attributes of vm.'''
    def __init__(self, attributes):
        # attributes in machine_attributes.json
        self.id = attributes['id']
        self.task_type = attributes['task_type']
        self.location = attributes['location']
        self.cr = attributes['cpu_capacity']
        self.price = attributes['price']
        self.origin_price = attributes['price']
        self.local_bw_up = 100000 # Kbps
        self.local_bw_down = 100000 # Kbps
        # the bw and daley to user (runtime)
        self.from_user = {}
        # the average bw of all user (runtime)
        self.avg_bw_up = None
        self.avg_bw_down = None