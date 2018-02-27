# -*- coding: utf-8 -*-
"""
Loss functions for heteroscedatic regression
1) L2Loss with noise --> p(y|x) ~ Normal(f(x),sigma(x))
2) L1Loss with noise --> p(y|x) ~ Laplacian(f(x),sigma(x))
"""
from __future__ import absolute_import, print_function, division

import tensorflow as tf

from niftynet.engine.application_factory import LossHeteroRegressionFactory
from niftynet.layer.base_layer import Layer


class LossFunction(Layer):
    def __init__(self,
                 loss_type='L2Loss',
                 loss_func_params=None,
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)

        # set loss function and function-specific additional params.
        self._data_loss_func = LossHeteroRegressionFactory.create(loss_type)
        self._loss_func_params = \
            loss_func_params if loss_func_params is not None else {}

    def layer_op(self,
                 prediction,
                 noise,
                 ground_truth=None,
                 weight_map=None):
        """
        Compute learned loss attenuation from ``prediction`` and ``ground truth``,
        weighted by the heteroscedatic ``noise``.
        The computed loss map are weighted by ``weight_map``.

        The ``noise`` is learned by considering s := log(sigma^2) for numerical
        reasons

        if ``prediction`` is list of tensors, each element of the list
        will be compared against ``ground_truth` and the weighted by
        ``weight_map``.

        :param prediction: input will be reshaped into
            ``(batch_size, N_voxels, num_classes)``
        :param ground_truth: input will be reshaped into
            ``(batch_size, N_voxels)``
        :param noise: input will be reshaped into
            ``(batch_size, N_voxels)``
        :param weight_map: input will be reshaped into
            ``(batch_size, N_voxels)``
        :return:
        """

        with tf.device('/cpu:0'):
            batch_size = ground_truth.shape[0].value
            ground_truth = tf.reshape(ground_truth, [batch_size, -1])
            noise = tf.reshape(noise, [batch_size, -1])
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
                    noise_b = noise[b_ind]
                    ground_truth_b = ground_truth[b_ind]
                    weight_b = None if weight_map is None else weight_map[b_ind]

                    loss_params = {
                        'prediction': pred_b,
                        'ground_truth': ground_truth_b,
                        'noise': noise_b,
                        'weight_map': weight_b}
                    if self._loss_func_params:
                        loss_params.update(self._loss_func_params)

                    loss_batch.append(self._data_loss_func(**loss_params))
                data_loss.append(tf.reduce_mean(loss_batch))
            return tf.reduce_mean(data_loss)


def l1_loss(prediction, ground_truth, noise, weight_map=None):
    """
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :return: mean of the l1 loss across all voxels.
    """

    precision = 0.5*tf.exp(-noise)

    absolute_residuals = tf.abs(tf.subtract(prediction, ground_truth))

    #if weight_map is not None:
    #    absolute_residuals = tf.multiply(absolute_residuals, weight_map)
    #    sum_residuals = tf.reduce_sum(absolute_residuals)
    #    sum_weights = tf.reduce_sum(weight_map)

    sum_residuals = tf.reduce_sum(absolute_residuals)
    sum_weights = tf.size(absolute_residuals)

    loss = tf.multiply(absolute_residuals, precision)
    loss = tf.add(loss, noise)

    return tf.reduce_mean(tf.truediv(tf.cast(loss, dtype=tf.float32),
                          tf.cast(sum_weights, dtype=tf.float32)))


def l2_loss(prediction, ground_truth, noise, weight_map=None):
    """
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :return: 1/numVox * sum_numVox (1/2)exp(-s) * L2loss + s
    """

    # noise_regulariser is log(sigma^2) and not 1/2log(sigma^2)
    # why? to be consistent with hetero seg due to Kendall approximation

    # noise is net_out
    # assume noise = log(sigma)

    # From Gal et al. NIPS 2017:
    # Equation (8) is: (1/2)*exp(-s)||y - pred||^2 + (1/2)
    # where s := log(sigma^2) so sigma = sqrt(exp(s))

    small_constant = 1e-6
    sigma_opt = tf.square(tf.exp(noise) + small_constant)

    residuals = tf.subtract(prediction, ground_truth)
    squared_residuals = tf.square(residuals)

    loss = (1/sigma_opt) * squared_residuals + tf.log(sigma_opt)
    return tf.reduce_mean(loss)


def l2_loss_img(prediction, ground_truth, noise, weight_map=None):
    """
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :return: loss image to weight voxel wise in hetero multi-task loss
    """

    precision = 0.5*tf.exp(-noise)
    residuals = tf.subtract(prediction, ground_truth)

    if weight_map is not None:
        residuals = \
            tf.multiply(residuals, weight_map) / tf.reduce_sum(weight_map)

    squared_residuals = tf.square(residuals)
    loss = tf.add(tf.multiply(precision, squared_residuals), noise)

    return loss


def l2_outlier_loss(prediction, ground_truth, noise, weight_map=None):
    """

    Likelihood = N(mu, sigma) + U(range)

    Help deal with outliers

    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :return: 1/numVox * sum_numVox (1/2)exp(-s) * L2loss + s + Uniform
    """
