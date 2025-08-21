import os
from pathlib import Path

import numpy as np 

import allensdk

# has our smoothing functions
from swdb_utils import *

import gc
import warnings
warnings.filterwarnings("ignore") 


# import behavior project cache class to load the data
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

cache_dir = Path('/root/capsule/data/')
scratch_dir = Path('/root/capsule/scratch/')

cache = VisualBehaviorOphysProjectCache.from_local_cache(
            cache_dir=cache_dir, use_static_cache=True)


behavior_sessions = cache.get_behavior_session_table()


project = 'VisualBehavior'
exclude_session_type = 'TRAINING_0_gratings_autorewards_15min'

filtered_behavior_sessions = behavior_sessions[(behavior_sessions.project_code == project)&(behavior_sessions.session_type!=exclude_session_type)] 

behavior_session_ids = filtered_behavior_sessions.index.values



all_smoothed_run_speeds = []
all_smoothed_timestamps = []

# Let's get a random sample to explore
number_of_sessions = 100
random_sessions = np.random.choice(behavior_session_ids,number_of_sessions,replace=False)

for behavior_session_id in random_sessions:
    
    behavior_session = cache.get_behavior_session(behavior_session_id)
    
    run_df = behavior_session.running_speed

    run_speed = run_df.speed.values
    run_times = run_df.timestamps.values
    
    # let's filter the data to only change detection blocks for consistency
    stimulus_table = behavior_session.stimulus_presentations
    task_df = stimulus_table[stimulus_table.stimulus_block_name.isin(['change_detection_behavior','change_detection_passive'])]
    t0,tf = task_df.start_time.values[0], task_df.end_time.values[-1]
    
    filter_cond = (run_times>=t0)&(run_times<=tf)
    task_run_speed = run_speed[filter_cond]
    task_run_times = run_times[filter_cond]
    
    smoothed_run_speed, smoothed_run_times = apply_sliding_window_average_to_timeseries(input_times=task_run_times,
                                                                             input_variable=task_run_speed,
                                                                             window_size_t=30,
                                                                             step_size_t=15,
                                                                             align_time='center')
    
    all_smoothed_run_speeds.append(smoothed_run_speed)
    all_smoothed_timestamps.append(smoothed_run_times)
    
    
    # getting so many datasets can be memory intensive, here's a simple way of saving memory
    session_clean_up = [behavior_session,run_df,task_df,run_speed,run_times,
                        filter_cond,task_run_speed,task_run_times,stimulus_table]
    
    # delete anything we won't be saving
    for trash in session_clean_up:
        del trash 
    gc.collect() # reclaim memory using garbage collector


# traces may be different lengths despite having the same dt (step_size_t)
min_dur = np.inf

for trace in all_smoothed_run_speeds:
    if len(trace)<min_dur:
        min_dur = len(trace)
        
num_sessions = len(all_smoothed_run_speeds)        
running_mat = np.zeros((num_sessions,min_dur))
timestamp_mat = np.zeros((num_sessions,min_dur))

# standardize the duration for all sessions and set initial time point to 0
for k in range(num_sessions):
    running_mat[k,:] = all_smoothed_run_speeds[k][:min_dur]
    timestamp_mat[k,:] = all_smoothed_timestamps[k][:min_dur] - all_smoothed_timestamps[k][0] # shift by first time point
    

# save the matrices
path_to_datacache = scratch_dir / 'behavior'
if not path_to_datacache.is_dir(): 
    path_to_datacache.mkdir(parents=True,exist_ok=True)
    
    
filename = 'smoothed_running_traces.npy'
f = path_to_datacache / filename
np.save(f,running_mat)
    
filename = 'smoothed_timestamps.npy'
f = path_to_datacache / filename
np.save(f,timestamp_mat)

filename = 'smoothed_behavior_session_ids.npy'
f = path_to_datacache / filename
np.save(f,random_sessions)