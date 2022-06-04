import numpy as np
import scipy.integrate as integrate
import json
import os

np.random.seed(1126)

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
total_time = 30000
history_time = 5000
dir = 'data/case4/'
# spe: seconds per event arrive
voip_spe = 1 / 30
ipVideo_spe = 1 / 30
ftp_spe = 1 / 100

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
        attrs = (("VoIP", 0.3, (17, 13, 300, 200), (4, 4, 3, 15)), ("IP_Video", 0.4, (2, 4, 45, 5), (4, 4, 500, 700)),
                ("FTP", 0.7, (5, 3, 40, 10), (4, 5, 700, 900)))
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

machine_generator(dir + 'machine_attributes.json')
# runtime task
task_events_generator(dir + 'task_events.json', total_time)
# history data
task_events_generator(dir + 'history_data.json', history_time)