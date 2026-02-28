import pm4py
from pm4py.objects.log.obj import EventLog, Trace
import pandas as pd
import numpy as np


def import_log(file_path: str) -> EventLog:
    # Imports an XES file and converts it to EventLog

    log = pm4py.read_xes(file_path)
    
    if not isinstance(log, EventLog):
        from pm4py.objects.conversion.log import converter as log_converter
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    
    return log


def create_prefixes_log(log: EventLog, prefix_length: int, use_padding: bool = True) -> EventLog:
    # Creates a new log containing only the first 'prefix_length' events of each trace
 
    from pm4py.objects.log.obj import Event
    
    prefixes_log = EventLog()
    
    for trace in log:
        prefix_trace = Trace()
        
        prefix_trace.attributes.update(trace.attributes)
        
        for event in trace[:prefix_length]:
            prefix_trace.append(event)
        # Add padding events if the trace is shorter than prefix_length
        if use_padding and len(prefix_trace) < prefix_length:
            padding_needed = prefix_length - len(prefix_trace)
            for i in range(padding_needed):
                padding_event = Event()
                padding_event['concept:name'] = 'PADDING'
                padding_event['lifecycle:transition'] = 'complete'
                prefix_trace.append(padding_event)
        
        prefixes_log.append(prefix_trace)
    
    return prefixes_log


def get_activity_names(log: EventLog) -> list[str]:
    # Extracts all unique activity names present in the log

    activity_names = []
    
    for trace in log:
        for event in trace:
            activity_name = event['concept:name']
            if activity_name not in activity_names:
                activity_names.append(activity_name)
    
    return activity_names


def boolean_encode(log: EventLog, activity_names: list[str]) -> pd.DataFrame:
    # Encodes the EventLog into a DataFrame with boolean features for each activity and prefix_length

    encoded_data = []
    
    for trace in log:
        row = {
            'trace_id': trace.attributes.get('concept:name', 'unknown'),
            'prefix_length': len(trace)
        }
        
        for activity in activity_names:
            row[activity] = False
        
        for event in trace:
            activity = event['concept:name']
            row[activity] = True
        
        label = trace.attributes.get('label', False)
        if isinstance(label, str):
            row['label'] = label
        else:
            row['label'] = 'true' if label else 'false'
        
        encoded_data.append(row)
    
    columns = ['trace_id', 'prefix_length'] + activity_names + ['label']
    df = pd.DataFrame(encoded_data, columns=columns)
    
    return df


def exclude_keys_from_trace(trace_dict: dict, keys_to_exclude: set = {'trace_id', 'label', 'prefix_length'}, remove_false_activities: bool = True) -> dict:
    # Excludes specific keys from a trace dictionary

    filtered = {}
    for k, v in trace_dict.items():
        if k in keys_to_exclude:
            continue
        if remove_false_activities and v != True:
            continue
        filtered[k] = v
    
    return filtered
