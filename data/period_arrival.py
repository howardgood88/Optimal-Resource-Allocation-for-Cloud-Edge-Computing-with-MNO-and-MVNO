import poisson_arrival
import os
import json
import numpy as np

# the traffic ratio over peek traffic of 24 hours in a day
hour_traffic_ratio = [0.51, 0.42, 0.33, 0.31, 0.23, 0.23, 0.24, 0.22, 0.24, 0.33, 0.35, 0.52, 0.56, 0.56, 0.64, 0.8, 0.91, 0.97, 0.98, 0.95, 0.92, 0.965, 0.87, 0.8]

machine_num = 300
user_num = 100
total_time = 30000
history_time = 5000
dir = 'data/case5/'
# spe: seconds per event arrive
voip_spe = 1 / 30
ipVideo_spe = 1 / 30
ftp_spe = 1 / 100

def task_events_generator(filename):
    def event_gen(_type, max_cpu, bw_up_attr, bw_down_attr):
        cpu_req = np.random.random() * max_cpu
        return [event_id, 0, t, _type, str(np.random.randint(0, user_num)), cpu_req, cpu_req * np.random.random(),
                poisson_arrival.beta(*bw_up_attr), poisson_arrival.beta(*bw_down_attr)]
    t = 0
    event_id = 0
    event_in = {}
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'a') as f:
        f.write('[\n')
        for hour_ratio in hour_traffic_ratio:
            freqs = (voip_spe * hour_ratio, ipVideo_spe * hour_ratio, ftp_spe * hour_ratio)
            attrs = (("VoIP", 0.3, (17, 13, 300, 200), (4, 4, 3, 15)), ("IP_Video", 0.4, (2, 4, 45, 5), (4, 4, 500, 700)),
                    ("FTP", 0.7, (5, 3, 40, 10), (4, 5, 700, 900)))
            while 1:
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
                if t % 3600 == 0:
                    break
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

poisson_arrival.machine_generator(dir + 'machine_attributes.json')
task_events_generator(dir + 'task_events.json')
poisson_arrival.task_events_generator(dir + 'history_data.json', 10000)