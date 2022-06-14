from poisson_arrival import beta
import os
import json
from parameters import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(rnd_seed)

# the traffic ratio over peek traffic of 24 hours in a day
# hour_traffic_ratio = [0.51, 0.42, 0.33, 0.31, 0.23, 0.23, 0.24, 0.22, 0.24, 0.33, 0.35, 0.52, 0.56, 0.56, 0.64, 0.8, 0.91, 0.97, 0.98, 0.95, 0.92, 0.965, 0.87, 0.8]
day_hour_traffic_ratio = [0.09, 0.09, 0.07, 0.06, 0.05, 0.09, 0.2, 0.6, 0.69, 0.67, 0.66, 0.5, 0.53, 0.6, 0.57, 0.41, 0.21, 0.19, 0.17, 0.18, 0.11, 0.09, 0.07, 0.08]
history_hour_traffic_ratio = [sum(day_hour_traffic_ratio) / len(day_hour_traffic_ratio) for _ in range(24)]
# print(f'history ratio: {history_hour_traffic_ratio[0]}')

user_num = 300
machine_num = 300
out_files = ['./data/case4/', './baselines/VM Load Balance/data/case4/', './baselines/Random/data/case4/']
number_of_days = 7

def machine_generator(filename):
    id = 1
    _str = '{\n'
    for id in range(machine_num):
        _type = np.random.choice(['VoIP', 'IP_Video', 'FTP'])
        location = np.random.choice(['cloud', 'edge'])
        discount = 1
        if location == 'edge':
            discount = 0.8
        cpu = np.random.random()
        price = int(cpu * 250 * discount)
        _str += f'"{id}":{{"id":"{id}","task_type":"{_type}","location":"{location}","cpu_capacity":{cpu:.2f},"price":{price}}}'
        if id != machine_num - 1:
            _str += ',\n'
        else:
            _str += '\n'
    _str += '}'
    # save to each dir
    for dir in out_files:
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + filename, 'w') as f:
            f.write(_str)

def task_events_generator(filename, hour_traffic_ratio):
    def event_gen(_type, max_cpu, bw_up_attr, bw_down_attr):
        cpu_req = np.random.random() * max_cpu
        return [event_id, 0, t, _type, str(np.random.randint(0, user_num)), cpu_req, cpu_req * np.random.random(),
                beta(*bw_up_attr), beta(*bw_down_attr)]
    t = 0
    event_id = 0
    event_in = {}
    
    _str = '[\n'
    for i in range(number_of_days):
        if i % 7 < 5:
            r = 1
        else:
            r = 0.12
        for hour_ratio in hour_traffic_ratio:
            freqs = (voip_spe * hour_ratio * r, ipVideo_spe * hour_ratio * r, ftp_spe * hour_ratio * r)
            attrs = (("VoIP", 0.3, voip_bw_up_attr, voip_bw_down_attr), ("IP_Video", 0.4, ipVideo_bw_up_attr, ipVideo_bw_down_attr),
                    ("FTP", 0.7, ftp_bw_up_attr, ftp_bw_down_attr))
            while 1:
                # outcome events
                if t in event_in:
                    for event in event_in[t]:
                        event[1] = 1
                        event[2] = t
                        _str += json.dumps(event) + ',\n'
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
                        _str += json.dumps(event) + ',\n'
                        event_id += 1
                t += 1
                if t % 3600 == 0:
                    break
    for t, events in sorted(event_in.items(), key=lambda x: x[1]):
        for event in events:
            event[1] = 1
            event[2] = t
            _str += json.dumps(event) + ',\n'
            event_id += 1
    # delete the last ',' to make a valid list
    _str = _str[:-2] + '\n]'
    
    # save to each dir
    for dir in out_files:
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(dir + filename, 'w') as f:
            f.write(_str)

if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.makedirs('./data')
    plt.figure()
    plt.plot(np.arange(len(day_hour_traffic_ratio)), day_hour_traffic_ratio)
    plt.savefig('data/daily_pattern')
    
    machine_generator('machine_attributes.json')
    task_events_generator('task_events.json', day_hour_traffic_ratio)
    task_events_generator('history_data.json', history_hour_traffic_ratio)
    print(f'Finished generating, save result to {out_files}')