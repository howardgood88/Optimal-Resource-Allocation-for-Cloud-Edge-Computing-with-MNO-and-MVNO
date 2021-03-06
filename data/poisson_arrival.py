import numpy as np
import scipy.integrate as integrate
from parameters import *
import json
import os

np.random.seed(rnd_seed)

def beta(a, b, t, d):
    '''Beta Distribution Generator.'''
    def distribution(x):
        return ((x - d) / t) ** (a - 1) * (1 - (x - d) / t) ** (b - 1) / integrate.quad(lambda x: x ** (a - 1) * (1 - x) ** (b - 1), 0, 1)[0]

    mode = (a - 1) / (a + b - 2)
    max_val = distribution(mode * t + d)
    while True:
        x = np.random.uniform(d, d + t)
        y = np.random.uniform(0, max_val)
        if y <= distribution(x):
            return x # Kbps

machine_num = 300
user_num = 100
total_time = 3600 * 6
history_time = 3600 * 1
dir = './data/case1/'

def machine_generator(filename):
    id = 1
    with open(filename, 'w') as f:
        f.write('{\n')
        for id in range(machine_num):
            _type = np.random.choice(['VoIP', 'IP_Video', 'FTP'])
            location = np.random.choice(['cloud', 'edge'])
            discount = 1
            if location == 'edge':
                discount = 0.8
            cpu = np.random.random()
            price = int(cpu * 250 * discount)
            f.write(f'"{id}":{{"id":"{id}","task_type":"{_type}","location":"{location}","cpu_capacity":{cpu:.2f},"price":{price}}}')
            if id != machine_num - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write('}')

def task_events_generator(filename, last_t):
    def event_gen(_type, max_cpu, bw_up_attr, bw_down_attr):
        cpu_req = np.random.random() * max_cpu
        return [event_id, 0, t, _type, str(np.random.randint(0, user_num)), cpu_req, cpu_req * np.random.random(),
                beta(*bw_up_attr), beta(*bw_down_attr)]
    t = 0
    event_id = 0
    event_in = {}
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'a') as f:
        f.write('[\n')
        freqs = (voip_spe, ipVideo_spe, ftp_spe)
        attrs = (("VoIP", 0.3, voip_bw_up_attr, voip_bw_down_attr), ("IP_Video", 0.4, ipVideo_bw_up_attr, ipVideo_bw_down_attr),
                    ("FTP", 0.7, ftp_bw_up_attr, ftp_bw_down_attr))
        while t < last_t:
            # outcome events
            if t in event_in:
                for event in event_in[t]:
                    event[1] = 1
                    event[2] = t
                    f.write(json.dumps(event) + ',\n')
                    event_id += 1
                del event_in[t]
            # income events
            for freq, attr in zip(freqs, attrs):
                for _ in range(np.random.poisson(freq)):
                    event = event_gen(*attr)
                    interval = min(np.random.randint(1, 1000), 300) # maximum interval in google dataset is 5min
                    end_t = t + interval
                    if end_t not in event_in:
                        event_in[end_t] = []
                    event_in[end_t].append(event)
                    f.write(json.dumps(event) + ',\n')
                    event_id += 1
            t += 1
        for t, events in sorted(event_in.items(), key=lambda x: x[1]):
            for event in events:
                event[1] = 1
                event[2] = t
                f.write(json.dumps(event) + ',\n')
                event_id += 1
        # delete the ',' of the last line
        f.seek(f.tell() - 3, 0)
        f.truncate()
        f.write('\n]')

if __name__ == '__main__':
    machine_generator(dir + 'machine_attributes.json')
    # runtime task
    task_events_generator(dir + 'task_events.json', total_time)
    # history data
    task_events_generator(dir + 'history_data.json', history_time)
    print(f'Finished generating, save result to {dir}')