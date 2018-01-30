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

        # set loss function and function-specific additional params.
        self._data_loss_func = LossMultiTaskFactory.create(loss_type)
        self._loss_func_params = \
            loss_func_params if loss_func_params is not None else dict()

    def layer_op(self,
                 prediction,
                 ground_truth=None,
                 weight_map=None):
        """
        Compute loss from ``prediction`` and ``ground truth``,
        the computed loss map are weighted by ``weight_map``.

        if ``prediction`` is list of tensors, each element of the list
        will be compared against ``ground_truth` and the weighted by
        ``weight_map``.

        :param prediction: input will be reshaped into
            ``(batch_size, N_voxels, num_classes)``
        :param ground_truth: input will be reshaped into
            ``(batch_size, N_voxels)``
        :param weight_map: input will be reshaped into
            ``(batch_size, N_voxels)``
        :return:
        """

        with tf.device('/cpu:0'):
            batch_size = ground_truth.shape[0].value
            ground_truth = tf.reshape(ground_truth, [batch_size, -1])
            if weight_map is not None:
                weight_map = tf.reshape(weight_map, [batch_size, -1])
            if not isinstance(prediction, (list, tuple)):
                prediction = [prediction]

            data_loss = []
            for ind, pred in enumerate(prediction):
                # go through each scale

                loss_batch = []
                for b_ind, pred_b in enumerate(tf.unstack(pred, axis=0)):
                    # go through each image in a batch

                    pred_b = tf.reshape(pred_b, [-1])
                    ground_truth_b = ground_truth[b_ind]
                    weight_b = None if weight_map is None else weight_map[b_ind]

                    loss_params = {
                        'prediction': pred_b,
                        'ground_truth': ground_truth_b,
                        'weight_map': weight_b}
                    if self._loss_func_params:
                        loss_params.update(self._loss_func_params)

                    loss_batch.append(self._data_loss_func(**loss_params))
                data_loss.append(tf.reduce_mean(loss_batch))
            return tf.reduce_mean(data_loss)


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




















