import numpy as np


def sliding_window(timestamps, window_size, step_size):
    """
    Generates sliding windows from a sequence with a specified step size.

    Args:
        timestamps (list or str): Timestamps associated with input variable (e.g., list, string).
        window_size (int): The size of each sliding window.
        step_size (int): The number of elements to advance the window by in each step.

    Yields:
        list or str: A sub-sequence representing the current window.
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if not isinstance(step_size, int) or step_size <= 0:
        raise ValueError("step_size must be a positive integer.")
    if window_size > len(timestamps):
        raise ValueError("window_size cannot be greater than the sequence length.")

    for i in range(0, len(timestamps) - window_size + 1, step_size):
        yield timestamps[i:i + window_size]
        
        
def apply_sliding_window_average_to_timeseries(input_times,input_variable,window_size_t=60,step_size_t=30,align_time='center'):
    """
    Applies sliding window across timeseries to temporally smooth and downsample data

    Args:
        input_times:
        input_variable:
        window_size_t:
        step_size_t: 
        align_time:

    Returns:
        mean_variable:
        window_times: 
    """
    dt = input_times[1]-input_times[0]
    
    window_size = int(window_size_t//dt) # secs -> samples
    step_size = int(step_size_t//dt)
    
    mean_variable = []
    window_times = []
    
    for window in sliding_window(input_times, window_size, step_size):
        t0,tf = window[0],window[-1]-dt
        
        mu = np.mean(input_variable[(input_times>=t0)&(input_times<=tf)])
        mean_variable.append(mu)
        
        if align_time == 'center':
            window_times.append((tf+t0)/2)
        elif align_time == 'right':
            window_times.append(tf)
        elif align_time == 'left':
            window_times.append(t0)  
        else:
            raise Exception("VariableError: %s is not defined for input 'align_time'"%align_time)
            
    return np.array(mean_variable),np.array(window_times)
    