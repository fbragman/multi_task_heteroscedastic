# -*- coding: utf-8 -*-
"""
Loss functions for multi-task regression and segmentation
"""

from __future__ import absolute_import, print_function, division

import tensorflow as tf

from niftynet.engine.application_factory import LossMultiTaskFactory
from niftynet.layer.base_layer import Layer


class LossFunction(Layer):
    def __init__(self,
                 loss_type='homoscedatic_1',
                 loss_func_params=None,
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)
        if loss_func_params is not None:
            self._loss_func_params = loss_func_params
        else:
            self._loss_func_params = {}
        self._data_loss_func = None
        self.make_callable_loss_func(loss_type)

    def make_callable_loss_func(self, type_str):
        self._data_loss_func = LossMultiTaskFactory.create(type_str)

    def layer_op(self):
        with tf.device('/cpu:0'):
            return self._data_loss_func(**self._loss_func_params)


def homoscedatic_loss_approx(loss_task_1, loss_task_2, sigma_1, sigma_2):
    """
    The multi-task loss with homoscedatic noise weighting as defined by
    Kendall et al. (2017) - Multi-Task Learning using Uncertainty to Weigh
    Losses for Scene Geometry and Semantics

    Equation 11 in paper, that uses an approximation to go from
    Softmax(W, sigma) to Softmax(W) + log(sigma)

    Note: s = log(sigma) is the value optimised for numerical stability

    :param loss_task_1: the current loss for task 1 (scalar)
    :param loss_task_2: the current loss for task 2 (scalar)
    :param sigma_1: homoscedatic noise estimation for task 1 (scalar)
    :param sigma_2: homoscedatic noise estimation for task 2 (scalar)
    :return:
    """

    task_1_precision = 2*tf.exp(-sigma_1)
    task_2_precision = 2*tf.exp(-sigma_2)

    task_1_weighted_loss = task_1_precision * loss_task_1 + sigma_1
    task_2_weighted_loss = task_2_precision * loss_task_2 + sigma_2
    total_loss = task_1_weighted_loss + task_2_weighted_loss

    return total_loss




















