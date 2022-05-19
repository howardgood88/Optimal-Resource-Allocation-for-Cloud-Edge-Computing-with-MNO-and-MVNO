import numpy as np
from parameters import (Task_event_index)

class Task_handler:
    task_events = None
    mask = None

    @classmethod
    def set_mask(cls, task_id):
        '''Set mask for method get_deleted_events and delete_events.'''
        cls.mask = np.where(cls.task_events[:, Task_event_index.index.value] == task_id)

    @classmethod
    def get_deleted_events(cls):
        return cls.task_events[cls.mask]

    @classmethod
    def delete_events(cls):
        return np.delete(cls.task_events, cls.mask)

    @classmethod
    def insert_event(cls, event):
        event_time_idx = Task_event_index.event_time.value
        insert_idx = np.where(cls.task_events[:, event_time_idx] >= event[event_time_idx])[0][0]
        return np.insert(cls.task_events, insert_idx, event, axis=0)