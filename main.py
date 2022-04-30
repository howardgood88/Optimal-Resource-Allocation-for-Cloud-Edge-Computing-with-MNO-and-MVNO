import json

def loadData(filename: str) -> dict:
    '''Load data from filename'''

    with open(filename, 'r') as f:
        data = json.load(f)
    return data

dir = './data/case1/'
machine_attributes = loadData(dir + 'machine_attributes.json')
task_events = loadData(dir + 'task_events.json')