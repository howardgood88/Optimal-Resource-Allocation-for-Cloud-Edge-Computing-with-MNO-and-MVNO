import numpy as np
from parameters import (Task_event_index)

class Task_handler:
    task_events = None
    mask = None
    changed = False

    @classmethod
    def set_mask(cls, task_id):
        '''Set mask for method get_deleted_events and delete_events.'''
        cls.mask = np.where(cls.task_events[:, Task_event_index.index.value] == task_id)

    @classmethod
    def get_deleted_events(cls):
        '''Get the two events under mask.'''
        cls.changed = True
        return cls.task_events[cls.mask]

    @classmethod
    def delete_events(cls):
        '''Delete the two events under mask.'''
        cls.changed = True
        cls.task_events = np.delete(cls.task_events, cls.mask, axis=0)

    @classmethod
    def insert_event(cls, event):
        '''Insert the new event.'''
        cls.changed = True
        event_time_idx = Task_event_index.event_time.value
        wh = np.where(cls.task_events[:, event_time_idx] >= event[event_time_idx])[0]
        if wh.size == 0:
            wh = [len(cls.task_events)]
        insert_idx = wh[0]
        cls.task_events = np.insert(cls.task_events, insert_idx, event, axis=0)