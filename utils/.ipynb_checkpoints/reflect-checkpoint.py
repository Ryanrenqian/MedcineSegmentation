"""
Reflection of module
"""
import sys

sys.path.append('../')
from import basic
import pdb
from basic.model import *
from basic import *
from basic.eval_test import *
from basic import *
from basic import *

def get_package(package_name):
    return getattr(basic, package_name)


def get_module(package_name, module_name):
    _package = get_package(package_name)
    return getattr(_package, module_name)


def get_class(package_name, module_name, class_name):
    _module = get_module(package_name, module_name)
    return getattr(_module, class_name)


def get_model(config):
    """获取模型"""
    _model_module = get_module('model', config.get_config('model', 'model_module'))
    return getattr(_model_module, 'load_model')(config.get_config('model', 'model_function'),
                                                config.get_config('model', 'model_name'))


def get_instance(config, config_key):
    """
    获取 train test validate
    """
    _Class = get_class('eval_' + config_key, config.get_config(config_key, 'module_name'),
                       config.get_config(config_key, 'class_name'))
    return _Class(config)
