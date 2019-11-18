import os
import sys
import time
from basic.utils import timeutil
from basic.utils import file


class Log(object):
    """日志类
    单例模式 """
    _instance = None
    _lock = False

    def __new__(cls, log_path=None, console=True):
        if not cls._instance:
            cls._instance = object.__new__(cls)
            cls._instance.log_path = log_path
            cls._instance.console = console
            cls.lock = False
        return cls._instance

    def info(self, info, end='\n'):
        """输出日志，保护线程独立，简单lock即可"""
        while self._lock:
            pass
        self._lock = True

        log_f = open(self.log_path, 'a')
        if self.console:
            print(info, end=end)
        log_f.writelines('%s\n' % info)
        log_f.flush()
        log_f.close()

        self._lock = False
