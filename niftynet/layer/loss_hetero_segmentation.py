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
                 loss_type='scaled_approx_softmax',
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
                 weight_map=None,
                 T=None):
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

                    # preparing loss function parameters
                    loss_params = {
                        'prediction': pred_b,
                        'ground_truth': ground_truth_b,
                        'noise': noise_b,
                        'T': T,
                        'num_classes': self._num_classes}
                    if self._loss_func_params:
                        loss_params.update(self._loss_func_params)

                    # loss for each batch over spatial dimensions
                    loss_batch.append(self._data_loss_func(**loss_params))
                # loss averaged over batch
                data_loss.append(tf.reduce_mean(loss_batch))
            # loss averaged over multiple scales
            return tf.reduce_mean(data_loss)


def scaled_softmax(prediction, ground_truth, noise, T, num_classes):
    """
    Function to calculate the scaled cross-entropy loss function with likelihood function in form
    p(y|x,f(x),sigma) = Softmax((1/sigma^2)*f(x))

    Likelihood function analytically defined
    --> log p(y=c|f(x),sigma) = (1/sigma^2) * f(x)_c - log(sum_c'*exp((1/sigma^2)*f(x)_c'))

    To avoid dividing by zero (same as in regression - see loss_hetero_regression)
    set s := log(sigma^2), sigma = sqrt(exp(s))

    Equation (9) from Kendall et al. 2017
    (1/sigma^2) * f(x)_c - log(sum_c'*(1/sigma^2)*f(x)_c')
    --> exp(-s) * f(x)_c - log(sum_c'*(exp(exp(-s)*f(x)_c'))

    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground truth
    :param noise: modelled noise
    :param T: (not-used in this function)
    :return: the cross-entropy loss
    """

    class_mask = tf.one_hot(ground_truth, depth=num_classes)
    s = tf.exp(-noise)
    scaled_logit = tf.multiply(tf.expand_dims(s, 1), prediction)
    sm = tf.nn.log_softmax(scaled_logit)
    sm = tf.multiply(class_mask, sm)

    return tf.reduce_mean(sm)


def scaled_approx_softmax(prediction, ground_truth, noise, T, num_classes, weight_map=None):
    """
    Approximation to log-likelihood of scaled softmax as in Kendall
    Equation (10) from Kendall et a. 2017

    L = -log(softmax(f(x)) * (0.5 / sigma^2) + log(sigma^2)

    with s = log(sigma^2)

    L = -log(softmax(f(x)) * (0.5exp(-s)) + exp(-s)

    since -log(softmax) == cross-entropy

    we use entropy = sparse_softmax_cross_entropy_with_logits

    :param prediction: logits (before softmax)
    :param ground_truth: segmentation ground truth
    :param noise: modelled noise
    :param T: (not used)
    :param num_classes: number of classes
    :return:
    """

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=ground_truth)

    small_constant = 5e-02
    if small_constant > 0.:
        noise = tf.log(tf.exp(noise) + small_constant)

    precision = 0.5*(tf.exp(-noise))
    scaled_loss = tf.add(tf.multiply(precision, loss), noise)

    if weight_map is not None:
        weight_map = tf.cast(tf.size(scaled_loss), dtype=tf.float32) / \
                     tf.reduce_sum(weight_map) * weight_map
        scaled_loss = tf.multiply(scaled_loss, weight_map)
        print('DOING')

    return tf.reduce_mean(scaled_loss)


def scaled_approx_softmax_img(prediction, ground_truth, noise, T, num_classes):
    """
    Approximation to log-likelihood of scaled softmax as in Kendall
    Equation (10) from Kendall et a. 2017

    L = -log(softmax(f(x)) * (0.5 / sigma^2) + log(sigma^2)

    with s = log(sigma^2)

    L = -log(softmax(f(x)) * (0.5exp(-s)) + exp(-s)

    since -log(softmax) == cross-entropy

    we use entropy = sparse_softmax_cross_entropy_with_logits

    :param prediction: logits (before softmax)
    :param ground_truth: segmentation ground truth
    :param noise: modelled noise
    :param T: (not used)
    :param num_classes: number of classes
    :return: loss image
    """

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=ground_truth)


    small_constant = 1e-05

    precision = 2*tf.exp(noise + small_constant)
    scaled_loss = tf.add(tf.div(loss, precision), noise)

    return scaled_loss


def stoch_softmax(prediction, ground_truth, noise, T, num_classes):
    """
    Function to calculate the cross-entropy loss function with likelihood function in form
    p(y|x,f(x),sigma^2) = N(f(x),sigma^2)

    Requires stochastic sampling to calculationg the expectation of p

    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground truth
    :param noise: modelled noise
    :param T: stochastic passes to calculate expectation
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

    # binary mask for ground_truth

    class_mask = tf.one_hot(ground_truth, depth=num_classes)

    stochastic_loglik = tf.zeros(tf.shape(prediction))
    for _ in range(T):
        # random noise generation of size (batch_size, n_voxel, num_classes)
        random_noise = tf.random_normal(tf.shape(prediction), mean=0, stddev=1.0, dtype=tf.float32)

        # adding noise to the heteroscedatic noise map
        noise = tf.expand_dims(noise, 1)
        stochastic_logit = tf.add(prediction, noise * random_noise)

        # Calculate x - log(sum(exp(x))
        log_sm = tf.nn.log_softmax(stochastic_logit)

        # iterative addition of forward passes
        # add a mask since tf.nn.log_softmax = logits - log(reduce_sum(exp(logits), axis))
        # output shape = size of logits
        stochastic_loglik = tf.add(stochastic_loglik, tf.multiply(class_mask, log_sm))

    # calculate expectation
    expected_loglik = (1/T) * stochastic_loglik

    # log( expectation from stochastic passes )
    loss = tf.log(expected_loglik)

    # sum over all voxels
    return tf.reduce_sum(loss)




