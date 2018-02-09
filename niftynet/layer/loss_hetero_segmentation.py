# -*- coding: utf-8 -*-
"""
Loss functions for multi-class heteroscedatic segmentation
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.engine.application_factory import LossHeteroSegmentationFactory
from niftynet.layer.base_layer import Layer


class LossFunction(Layer):
    def __init__(self,
                 n_class,
                 loss_type='stoch_cross_entropy',
                 loss_func_params=None,
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)
        assert n_class > 0, \
            "Number of classes for segmentation loss should be positive."
        self._num_classes = n_class

        # set loss function and function-specific additional params.
        self._data_loss_func = LossHeteroSegmentationFactory.create(loss_type)
        self._loss_func_params = \
            loss_func_params if loss_func_params is not None else dict()

        if self._data_loss_func.__name__ == 'cross_entropy':
            tf.logging.info(
                'Cross entropy loss function calls '
                'tf.nn.sparse_softmax_cross_entropy_with_logits '
                'which always performs a softmax internally.')
            self._softmax = False

    def layer_op(self,
                 prediction,
                 noise,
                 ground_truth,
                 T,
                 weight_map=None):
        """
        Compute loss from `prediction` and `ground truth`,
        the computed loss map are weighted by `weight_map`.

        if `prediction `is list of tensors, each element of the list
        will be compared against `ground_truth` and the weighted by
        `weight_map`. (Assuming the same gt and weight across scales)

        :param prediction: input will be reshaped into
            ``(batch_size, N_voxels, num_classes)``
        :param ground_truth: input will be reshaped into
            ``(batch_size, N_voxels, ...)``
        :param noise: input will be reshaped into
            ``(batch_size, N_voxels, ...)``
        :param T: stochastic ``T`` passes to calculate expected log-likelihood
        :param weight_map: input will be reshaped into
            ``(batch_size, N_voxels, ...)``
        :return:
        """

        with tf.device('/cpu:0'):

            # prediction should be a list for multi-scale losses
            # single scale ``prediction`` is converted to ``[prediction]``
            if not isinstance(prediction, (list, tuple)):
                prediction = [prediction]

            data_loss = []
            for ind, pred in enumerate(prediction):
                # go through each scale

                loss_batch = []
                for b_ind, pred_b in enumerate(tf.unstack(pred, axis=0)):
                    # go through each image in a batch
                    pred_b = tf.reshape(pred_b, [-1, self._num_classes])

                    # reshape pred, ground_truth, weight_map to the same
                    # size: (n_voxels, num_classes)
                    # if the ground_truth has only one channel, the shape
                    # becomes: (n_voxels,)
                    spatial_shape = pred_b.shape.as_list()[:-1]
                    ref_shape = spatial_shape + [-1]
                    ground_truth_b = tf.reshape(ground_truth[b_ind], ref_shape)
                    noise_b = tf.reshape(noise[b_ind], ref_shape)
                    if ground_truth_b.shape.as_list()[-1] == 1:
                        ground_truth_b = tf.squeeze(ground_truth_b, axis=-1)
                    if noise_b.shape.as_list()[-1] == 1:
                        noise_b = tf.squeeze(noise_b, axis=-1)
                    if weight_map is not None:
                        weight_b = tf.reshape(weight_map[b_ind], ref_shape)
                        if weight_b.shape.as_list()[-1] == 1:
                            weight_b = tf.squeeze(weight_b, axis=-1)
                    else:
                        weight_b = None

                    # preparing loss function parameters
                    loss_params = {
                        'prediction': pred_b,
                        'ground_truth': ground_truth_b,
                        'noise': noise_b,
                        'stochastic_passes': T,
                        'weight_map': weight_b}
                    if self._loss_func_params:
                        loss_params.update(self._loss_func_params)

                    # loss for each batch over spatial dimensions
                    loss_batch.append(self._data_loss_func(**loss_params))
                # loss averaged over batch
                data_loss.append(tf.reduce_mean(loss_batch))
            # loss averaged over multiple scales
            return tf.reduce_mean(data_loss)


def stoch_cross_entropy(prediction, ground_truth, noise, T, weight_map=None):
    """
    Function to calculate the cross-entropy loss function

    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :return: the cross-entropy loss
    """

    # From Gal et al. NIPS 2017:
    # Equation (11) and (12)
    #
    # o. Perform T stochastic passes
    #
    # i. The logits are corrupted with Gaussian noise ~ N(0, I)
    #    where I is a diagonal matrix of size num_classes x num_classes
    #    x_{i,t} = logit_{i,t} + noise_{i} * N_{t}
    #
    # ii. L = sum_{i} log (1/T) sum_{t}(exp(x_{i,t,c} - log sum_{c'} exp(x_{i,t,c'})
    #
    # where i is the pixel, c is the class, c' sum of all classes and t is the stochastic pass
    #
    # noise generated on a voxel-wise basis with random samples from same distribution across logit classes

    stochastic_logliks = []
    for _ in range(T):
        # random noise generation of size (batch_size, n_voxel, num_classes)
        random_noise = tf.random_normal(prediction.shape, mean=0, stddev=1.0, dtype=tf.float32)

        # adding noise to the heteroscedatic noise map
        noise = tf.expand_dims(noise, 1)
        stochastic_logit = tf.add(prediction, noise * random_noise)

        # calculating log(softmax)
        stochastic_logliks.append(tf.nn.log_softmax(stochastic_logit))

    # calculate expectation
    expected_loglik = tf.reduce_mean(stochastic_logliks)

    # log( expectation from stochastic passes )
    loss = tf.log(expected_loglik)

    # sum over all voxels






    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=ground_truth)
    if weight_map is not None:
        weight_map = tf.cast(tf.size(entropy), dtype=tf.float32) / \
                     tf.reduce_sum(weight_map) * weight_map
        entropy = tf.multiply(entropy, weight_map)
    return tf.reduce_mean(entropy)




