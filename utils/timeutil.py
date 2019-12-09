import time


def get_timestr(time_stamp=None, str_format="%Y-%m-%d_%H-%M-%S"):
    if not time_stamp:
        time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    return time.strftime(str_format, local_time)

