import numpy as np

# spe: seconds per event
voip_spe = 5
ipVideo_spe = 3
ftp_spe = 10

t = 0
event_id = 0
event_in = {}
with open('task_events.json', 'w') as f:
    events = []
    for i in range(np.random.poisson(voip_spe)):
        if event_id not in event_in:
            event = [event_id, 0]
        else:
            event = [event_id, 1]
            del event_in[event_id]
        events.append(event)
    t += 1