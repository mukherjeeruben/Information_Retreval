import time
from datetime import datetime, timedelta
start_time = None
end_time = None


def time_stamp(stage):
    global start_time
    global end_time
    current_sys_time = datetime.now()
    current_sys_time = current_sys_time.strftime("%H:%M:%S")
    if stage == 'start':
        start_time = time.time()
        print('Start Time:', current_sys_time)
    else:
        end_time = time.time()
        total_time = time_difference(start_time, end_time)
        print('Total Execution Time:', total_time)
        print('End Time:', current_sys_time)
    return


def convert_time(time_sec):
    conversion = timedelta(seconds=time_sec)
    return str(conversion)


def time_difference(start, end):
    seconds_input = end - start
    total_time = convert_time(seconds_input)
    return total_time


