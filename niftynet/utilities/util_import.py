# -*- coding: utf-8 -*-
""" check module to be imported"""
import importlib

import tensorflow as tf


def check_module(name, min_version=None, descriptor='Optional', mandatory=False):
    """
    Check if the module exists, and
    satisfies the minimum version requirement.

    Raises ImportError and AssertionError.

    :param name:
    :param min_version:
    :param descriptor:
    :param mandatory:
    :return:
    """

    name = '{}'.format(name)
    if mandatory:
        log_level = tf.logging.fatal
    else:
        log_level = tf.logging.info

    try:
        the_module = importlib.import_module(name)
    except ImportError:
        log_level(
            descriptor + ' Python module %s not found, '
            'please install %s and retry if the application fails.',
            name, name)
        raise

    try:
        if min_version is not None:
            assert the_module.__version__ >= '{}'.format(min_version)
    except AttributeError:
        pass
    except AssertionError:
        log_level(
            descriptor + ' Python module %s version %s not found, '
            'please install %s-%s and retry if the application fails.',
            name, min_version, name, min_version)
        raise
