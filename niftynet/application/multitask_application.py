import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.engine.application_factory import InitializerFactory
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import NETWORK_OUTPUT
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.sampler_grid import GridSampler
from niftynet.engine.sampler_resize import ResizeSampler
from niftynet.engine.sampler_uniform import UniformSampler
from niftynet.engine.sampler_weighted import WeightedSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.layer.crop import CropLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.loss_multitask import LossFunction as LossFunction_MT
from niftynet.layer.loss_segmentation import LossFunction as LossFunction_Seg
from niftynet.layer.loss_regression import LossFunction as LossFunction_Reg
from niftynet.layer.loss_hetero_regression import LossFunction as LossFunction_HeteroReg
from niftynet.layer.loss_hetero_segmentation import LossFunction as LossFunction_HeteroSeg
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer

#TODO - make consistent with other new applications
#TODO - generalise to any multi-task

import numpy as np

SUPPORTED_INPUT = set(['image', 'output_1', 'output_2', 'weight', 'sampler'])


class MultiTaskApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = 'MULTITASK'

    # Initialise the class/application
    def __init__(self, net_param, action_param, is_training):
        BaseApplication.__init__(self)
        tf.logging.info('starting multi-task application')

        #self.is_training = is_training

        self.net_param = net_param
        self.action_param = action_param
        self.multitask_param = None

        self.data_param = None

        self.SUPPORTED_SAMPLING = {
            'uniform': (self.initialise_uniform_sampler,
                        self.initialise_grid_sampler,
                        self.initialise_grid_aggregator),
            'weighted': (self.initialise_weighted_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
            'resize': (self.initialise_resize_sampler,
                       self.initialise_resize_sampler,
                       self.initialise_resize_aggregator)
        }

    # Input data
    def initialise_dataset_loader(self, data_param=None, task_param=None, data_partitioner=None):

        self.data_param = data_param
        self.multitask_param = task_param

        # read each line of csv files into an instance of Subject
        if self.is_training:
            file_lists = []
            if self.action_param.validation_every_n > 0:
                file_lists.append(data_partitioner.train_files)
                file_lists.append(data_partitioner.validation_files)
            else:
                file_lists.append(data_partitioner.train_files)

            self.readers = []
            for file_list in file_lists:
                reader = ImageReader(SUPPORTED_INPUT)
                reader.initialise(data_param, task_param, file_list)
                self.readers.append(reader)
        else:
            inference_reader = ImageReader(['image'])
            file_list = data_partitioner.inference_files
            inference_reader.initialise(data_param, task_param, file_list)
            self.readers = [inference_reader]

        mean_var_normaliser = MeanVarNormalisationLayer(
            image_name='image')
        histogram_normaliser = None
        if self.net_param.histogram_ref_file:
            histogram_normaliser = HistogramNormalisationLayer(
                image_name='image',
                modalities=vars(task_param).get('image'),
                model_filename=self.net_param.histogram_ref_file,
                norm_type=self.net_param.norm_type,
                cutoff=self.net_param.cutoff,
                name='hist_norm_layer')

        label_normaliser = None
        if self.net_param.histogram_ref_file:
            label_normaliser = DiscreteLabelNormalisationLayer(
                image_name='label',
                modalities=vars(task_param).get('label'),
                model_filename=self.net_param.histogram_ref_file)

        normalisation_layers = []
        if self.net_param.normalisation:
            normalisation_layers.append(histogram_normaliser)
        if self.net_param.whitening:
            normalisation_layers.append(mean_var_normaliser)
        if task_param.label_normalisation and \
                (self.is_training or not task_param.output_prob):
            normalisation_layers.append(label_normaliser)

        augmentation_layers = []
        if self.is_training:
            if self.action_param.random_flipping_axes != -1:
                augmentation_layers.append(RandomFlipLayer(
                    flip_axes=self.action_param.random_flipping_axes))
            if self.action_param.scaling_percentage:
                augmentation_layers.append(RandomSpatialScalingLayer(
                    min_percentage=self.action_param.scaling_percentage[0],
                    max_percentage=self.action_param.scaling_percentage[1]))
            if self.action_param.rotation_angle:
                augmentation_layers.append(RandomRotationLayer())
                augmentation_layers[-1].init_uniform_angle(
                    self.action_param.rotation_angle)

        volume_padding_layer = []
        if self.net_param.volume_padding_size:
            volume_padding_layer.append(PadLayer(
                image_name=SUPPORTED_INPUT,
                border=self.net_param.volume_padding_size))
        for reader in self.readers:
            reader.add_preprocessing_layers(volume_padding_layer +
                                            normalisation_layers +
                                            augmentation_layers)

    def initialise_uniform_sampler(self):
        self.sampler = [[UniformSampler(
            reader=reader,
            data_param=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_weighted_sampler(self):
        self.sampler = [[WeightedSampler(
            reader=reader,
            data_param=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_resize_sampler(self):
        self.sampler = [[ResizeSampler(
            reader=reader,
            data_param=self.data_param,
            batch_size=self.net_param.batch_size,
            shuffle_buffer=self.is_training,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_sampler(self):
        self.sampler = [[GridSampler(
            reader=reader,
            data_param=self.data_param,
            batch_size=self.net_param.batch_size,
            spatial_window_size=self.action_param.spatial_window_size,
            window_border=self.action_param.border,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_aggregator(self):
        self.output_decoder = GridSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order)

    def initialise_resize_aggregator(self):
        self.output_decoder = ResizeSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order)

    def initialise_sampler(self):
        if self.is_training:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][0]()
        else:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][1]()

    def initialise_network(self):
        w_regularizer = None
        b_regularizer = None
        reg_type = self.net_param.reg_type.lower()
        decay = self.net_param.decay
        if reg_type == 'l2' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l2_regularizer(decay)
            b_regularizer = regularizers.l2_regularizer(decay)
        elif reg_type == 'l1' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l1_regularizer(decay)
            b_regularizer = regularizers.l1_regularizer(decay)

        self.net = ApplicationNetFactory.create(self.net_param.name)(
            num_classes=self.multitask_param.num_classes,
            dropout_rep=self.multitask_param.dropout_representation,
            w_initializer=InitializerFactory.get_initializer(
                name=self.net_param.weight_initializer),
            b_initializer=InitializerFactory.get_initializer(
                name=self.net_param.bias_initializer),
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):

        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()

        if self.is_training:
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(True),
                                    lambda: switch_sampler(False))
            else:
                data_dict = switch_sampler(for_training=True)

            image = tf.cast(data_dict['image'], tf.float32)
            net_out = self.net(image, is_training=self.is_training)

            if self.multitask_param.noise_model == 'homo':
                net_out_task_1 = net_out[0]
                net_out_task_2 = net_out[1]
                noise_out_task_1 = None
                noise_out_task_2 = None
            elif self.multitask_param.noise_model == 'hetero':
                net_out_task_1 = net_out[0]
                noise_out_task_1 = net_out[1]
                net_out_task_2 = net_out[2]
                noise_out_task_2 = net_out[3]
            elif self.multitask_param.noise_model == 'single-hetero':
                net_out_task_1 = net_out[0]
                noise_out_task_1 = net_out[1]
                net_out_task_2 = None
                noise_out_task_2 = None

            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)

            crop_layer = CropLayer(
                border=self.multitask_param.loss_border, name='crop-88')

            if self.multitask_param.noise_model == 'hetero':

                prediction_task_1 = crop_layer(net_out_task_1)
                prediction_task_2 = crop_layer(net_out_task_2)
                pred_noise_task_1 = crop_layer(noise_out_task_1)
                pred_noise_task_2 = crop_layer(noise_out_task_2)

            elif self.multitask_param.noise_model == 'homo':

                prediction_task_1 = crop_layer(net_out_task_1)
                prediction_task_2 = crop_layer(net_out_task_2)

            elif self.multitask_param.noise_model == 'single-hetero':

                prediction_task_1 = crop_layer(net_out_task_1)
                pred_noise_task_1 = crop_layer(noise_out_task_1)

            if self.multitask_param.noise_model == 'single-hetero':
                ground_truth_task_1 = crop_layer(data_dict.get('output_1', None))
            else:
                ground_truth_task_1 = crop_layer(data_dict.get('output_1', None))
                ground_truth_task_2 = crop_layer(data_dict.get('output_2', None))
                # Make sure ground truth is int32/int64
                ground_truth_task_2 = tf.to_int64(ground_truth_task_2)

            weight_map = None if data_dict.get('weight', None) is None \
                else crop_layer(data_dict.get('weight', None))

            data_loss_task_1, data_loss_task_2 = self.create_loss_functions(prediction_task_1,
                                                                            prediction_task_2,
                                                                            ground_truth_task_1,
                                                                            ground_truth_task_2,
                                                                            noise_task_1=pred_noise_task_1,
                                                                            noise_task_2=pred_noise_task_2)

            self.tensorboard_image_output_creator(outputs_collector,
                                                  prediction_task_1, prediction_task_2,
                                                  ground_truth_task_1, ground_truth_task_2,
                                                  noise_task_1=pred_noise_task_1, noise_task_2=pred_noise_task_2)

            # Set up the multi-task model
            # Note: if using hetero - only summed_loss should be really used
            #       homosecedatic_1 should only be used with noise_model = 'homo'
            multitask_loss = self.multitask_param.multitask_loss
            if self.multitask_param.noise_model != 'single-hetero':
                mt_loss_task = LossFunction_MT(multitask_loss)

            if self.multitask_param.noise_model == 'single-hetero':
                data_loss = data_loss_task_1

            else:

                if multitask_loss == 'summed_loss':
                    data_loss = mt_loss_task(data_loss_task_1, data_loss_task_2, None, None)
                elif multitask_loss == 'weighted_loss':
                    # weighted sum
                    w_1 = tf.get_variable('sigma_1',
                                          initializer=tf.constant(self.multitask_param.loss_sigma_1),
                                          trainable=False)
                    w_2 = tf.get_variable('sigma_2',
                                          initializer=tf.constant(self.multitask_param.loss_sigma_2),
                                          trainable=False)
                    data_loss = mt_loss_task(data_loss_task_1, data_loss_task_2, w_1, w_2)
                elif multitask_loss == 'homoscedatic_1':
                    w_1 = tf.get_variable('sigma_1',
                                          initializer=self.multitask_param.loss_sigma_1, trainable=True)
                    w_2 = tf.get_variable('sigma_2',
                                          initializer=self.multitask_param.loss_sigma_2, trainable=True)
                    data_loss = mt_loss_task(data_loss_task_1, data_loss_task_2, w_1, w_2)

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                loss = data_loss + reg_loss
            else:
                loss = data_loss

            grads = self.optimiser.compute_gradients(loss)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])

            # collecting output variables
            outputs_collector.add_to_collection(
                var=data_loss, name='Loss',
                average_over_devices=False, collection=CONSOLE)

            if multitask_loss == 'homoscedatic_1':
                # collecting output variables
                outputs_collector.add_to_collection(
                    var=tf.convert_to_tensor(w_1), name='w_1',
                    average_over_devices=False, collection=CONSOLE)

                # collecting output variables
                outputs_collector.add_to_collection(
                    var=tf.convert_to_tensor(w_2), name='w_2',
                    average_over_devices=False, collection=CONSOLE)

            if self.multitask_param.noise_model == 'hetero':
                # calculate the normal L2 and X-entropy to see...
                loss_func_task_1_val = LossFunction_Reg(loss_type='MAE')
                loss_func_task_2_val = LossFunction_Seg(n_class=self.multitask_param.num_classes[1],
                                                        loss_type='CrossEntropy')
                data_loss_task_1_val = loss_func_task_1_val(prediction=prediction_task_1,
                                                            ground_truth=ground_truth_task_1,
                                                            weight_map=weight_map)

                data_loss_task_2_val = loss_func_task_2_val(prediction=prediction_task_2,
                                                            ground_truth=ground_truth_task_2,
                                                            weight_map=weight_map)

                # output individual losses to Tensorboard
                outputs_collector.add_to_collection(
                    var=data_loss_task_1_val, name='Original_MAE',
                    average_over_devices=True, summary_type='scalar',
                    collection=TF_SUMMARIES)

                outputs_collector.add_to_collection(
                    var=data_loss_task_2_val, name='Original_cross_entropy',
                    average_over_devices=True, summary_type='scalar',
                    collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=data_loss, name='Loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            if self.multitask_param.noise_model == 'single-hetero':
                # output individual losses to Tensorboard
                if self.multitask_param.loss_1 == 'L2Loss':
                    Tloss_func_task_1 = LossFunction_Reg(loss_type='MAE')
                    Tdata_loss_task_1 = Tloss_func_task_1(prediction=prediction_task_1,
                                                        ground_truth=ground_truth_task_1,
                                                        weight_map=weight_map)
                elif self.multitask_param.loss_1 == 'ScaledApproxSoftMax':
                    Tloss_func_task_1 = LossFunction_Seg(n_class=self.multitask_param.num_classes[0],
                                                              loss_type='CrossEntropy')
                    Tdata_loss_task_1 = Tloss_func_task_1(prediction=prediction_task_1,
                                                        ground_truth=ground_truth_task_1)

                outputs_collector.add_to_collection(
                    var=Tdata_loss_task_1, name='Original',
                    average_over_devices=True, summary_type='scalar',
                    collection=TF_SUMMARIES)

            else:

                # output individual losses to Tensorboard
                outputs_collector.add_to_collection(
                    var=data_loss_task_1, name='Loss_Task_1',
                    average_over_devices=True, summary_type='scalar',
                    collection=TF_SUMMARIES)

                outputs_collector.add_to_collection(
                    var=data_loss_task_2, name='Loss_Task_2',
                    average_over_devices=True, summary_type='scalar',
                    collection=TF_SUMMARIES)

            if multitask_loss == 'homoscedatic_1':

                outputs_collector.add_to_collection(
                    var=tf.convert_to_tensor(w_1), name='sigma_1',
                    average_over_devices=True, summary_type='scalar',
                    collection=TF_SUMMARIES)

                outputs_collector.add_to_collection(
                    var=tf.convert_to_tensor(w_2), name='sigma_2',
                    average_over_devices=True, summary_type='scalar',
                    collection=TF_SUMMARIES)

        else:
            # Process regression and segmentation results individually
            # Segmentation --> convert logits to probabilities or argmax labels

            # Outputs hard-coded: net_out[0] - regression
            #                     net_out[1] - segmentation

            data_dict = switch_sampler(for_training=False)
            image = tf.cast(data_dict['image'], tf.float32)
            net_out = self.net(image, is_training=self.is_training)

            if self.multitask_param.noise_model == 'hetero':
                reg_out = net_out[0]
                reg_noise_out = (tf.exp(net_out[1])+0.005)
                seg_out = net_out[2]
                seg_noise_out = (tf.exp(net_out[3])+0.05)
            elif self.multitask_param.noise_model == 'homo':
                reg_out = net_out[0]
                seg_out = net_out[1]
            elif self.multitask_param.noise_model == 'single-hetero':
                if self.multitask_param.loss_1 == 'L2Loss':
                    reg_out = net_out[0]
                    noise_out = (tf.exp(net_out[1])+0.005)
                else:
                    seg_out = net_out[0]
                    noise_out = (tf.exp(net_out[1])+0.05)

            # Segmentation
            output_prob = self.multitask_param.output_prob
            num_classes_seg = self.multitask_param.num_classes[1]
            if self.multitask_param.noise_model == 'single-hetero':
                num_classes_seg = self.multitask_param.num_classes[0]
            if output_prob and num_classes_seg > 1:
                post_process_layer = PostProcessingLayer(
                    'SOFTMAX', num_classes=num_classes_seg)
            elif not output_prob and num_classes_seg > 1:
                post_process_layer = PostProcessingLayer(
                    'ARGMAX', num_classes=num_classes_seg)
            else:
                post_process_layer = PostProcessingLayer(
                    'IDENTITY', num_classes=num_classes_seg)

            if self.multitask_param.loss_1 == 'ScaledApproxSoftMax' \
                    or self.multitask_param.loss_2 == 'ScaledApproxSoftMax' \
                    or self.multitask_param.loss_2 == 'CrossEntropy':
                seg_out = post_process_layer(seg_out)

            crop_layer = CropLayer(border=0, name='crop-88')
            post_process_layer = PostProcessingLayer('IDENTITY')

            if 'reg_out' in locals():
                reg_out = post_process_layer(crop_layer(reg_out))

            if output_prob and num_classes_seg > 1:
                # softmax seg output

                # iterate over the classes and stack them all
                class_imgs = []
                for idx in range(num_classes_seg):
                    class_imgs.append(tf.expand_dims(seg_out[..., idx], -1))

                # Concatenation of tasks
                if self.multitask_param.noise_model == 'hetero':
                    # concat reg + reg_noise
                    mt_out = tf.concat([reg_out, reg_noise_out], -1)
                    # concat with seg probabilities
                    for idx in range(num_classes_seg):
                        mt_out = tf.concat([mt_out, class_imgs[idx]], -1)
                    # concat seg noise
                    mt_out = tf.concat([mt_out, seg_noise_out], -1)
                elif self.multitask_param.noise_model == 'homo':
                    for idx in range(num_classes_seg):
                        if idx == 0:
                            mt_out = tf.concat([reg_out, class_imgs[idx]], -1)
                        else:
                            mt_out = tf.concat([mt_out, class_imgs[idx]], -1)
                elif self.multitask_param.noise_model == 'single-hetero':
                    first = class_imgs
                    sec = noise_out
                    if type(first) is list:
                        mt_out = first[0]
                        for idx in np.linspace(1, num_classes_seg-1, num_classes_seg-1):
                            mt_out = tf.concat([mt_out, first[int(idx)]], -1)
                        mt_out = tf.concat([mt_out, sec], -1)
            else:
                # argmax output
                if self.multitask_param.noise_model == 'hetero':
                    mt_out = tf.stack([reg_out, reg_noise_out], axis=-1)
                    mt_out = tf.stack([mt_out, tf.cast(seg_out, tf.float32)], axis=-1)
                    mt_out = tf.stack([mt_out, seg_noise_out], axis=-1)
                elif self.multitask_param.noise_model == 'homo':
                    mt_out = tf.stack([reg_out, tf.cast(seg_out, tf.float32)], axis=-1)
                elif self.multitask_param.noise_model == 'single-hetero':
                    mt_out = tf.stack([reg_out, tf.sqrt(tf.exp(noise_out))], axis=-1)

            outputs_collector.add_to_collection(
                var=mt_out, name='window',
                average_over_devices=False, collection=NETWORK_OUTPUT)

            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)

            init_aggregator = \
                self.SUPPORTED_SAMPLING[self.net_param.window_sampling][2]
            init_aggregator()

    def tensorboard_image_output_creator(self, outputs_collector,
                                         prediction_task_1, prediction_task_2,
                                         ground_truth_task_1, ground_truth_task_2,
                                         noise_task_1=None, noise_task_2=None):
        """
        Output gt/prediction/noise to tensorboard IMAGE summaries
        :param prediction_task_1:
        :param prediction_task_2:
        :param ground_truth_task_1:
        :param ground_truth_task_2:
        :param noise_task_1:
        :param noise_task_2:
        :return:
        """

        def min_max_scaling(img):
            """
            Min-max scaling for output to 255-0 range
            :param img:
            :return:
            """
            return 255 * (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))

        def add_to_collector(img, name):
            """
            Add to collector
            :param img:
            :param name:
            :return:
            """
            import math
            outputs_collector.add_to_collection(
                var=tf.contrib.image.rotate(min_max_scaling(img), math.pi / 2),
                name=name, average_over_devices=True, summary_type='image',
                collection=TF_SUMMARIES)

        task_1 = self.multitask_param.task_1
        task_2 = self.multitask_param.task_2
        noise_model = self.multitask_param.noise_model

        if noise_model is 'homo':

            add_to_collector(prediction_task_1, task_1)
            add_to_collector(prediction_task_2, task_2)

            add_to_collector(ground_truth_task_1, task_1 + '_gt')
            add_to_collector(ground_truth_task_2, task_2 + '_gt')

        elif noise_model is 'hetero':

            add_to_collector(prediction_task_1, task_1)
            add_to_collector(noise_task_1, task_1 + '_noise')
            add_to_collector(prediction_task_2, task_2)
            add_to_collector(noise_task_2, task_2 + '_noise')

            add_to_collector(ground_truth_task_1, task_1 + '_gt')
            add_to_collector(ground_truth_task_2, task_2 + '_gt')

        elif noise_model is 'single-hetero':

            add_to_collector(prediction_task_1, task_1)
            add_to_collector(noise_task_1, task_1 + '_noise')

            add_to_collector(ground_truth_task_1, task_1 + '_gt')

    def create_loss_functions(self, prediction_task_1, prediction_task_2,
                                    ground_truth_task_1, ground_truth_task_2,
                                    noise_task_1=None, noise_task_2=None,
                                    weight_map=None):
        """
        Create loss functions to use
        :param prediction_task_1:
        :param prediction_task_2:
        :param ground_truth_task_1:
        :param ground_truth_task_2:
        :param noise_task_1:
        :param noise_task_2:
        :param weight_map:
        :return:
        """

        task_1 = self.multitask_param.task_1
        task_2 = self.multitask_param.task_2
        noise_model = self.multitask_param.noise_model

        # Set up the noise model
        if noise_model == 'homo' or noise_model is None:

            if task_1 is 'regression':
                loss_func_task_1 = LossFunction_Reg(loss_type=self.multitask_param.loss_1)
            elif task_1 is 'segmentation':
                loss_func_task_1 = LossFunction_Seg(n_class=self.multitask_param.num_classes[0],
                                                    loss_type=self.multitask_param.loss_1)
            if task_2 is 'regression':
                loss_func_task_2 = LossFunction_Reg(loss_type=self.multitask_param.loss_2)
            elif task_2 is 'segmentation':
                loss_func_task_2 = LossFunction_Seg(n_class=self.multitask_param.num_classes[1],
                                                    loss_type=self.multitask_param.loss_2)

            data_loss_task_1 = loss_func_task_1(prediction=prediction_task_1,
                                                ground_truth=ground_truth_task_1,
                                                weight_map=weight_map)

            data_loss_task_2 = loss_func_task_2(prediction=prediction_task_2,
                                                ground_truth=ground_truth_task_2,
                                                weight_map=weight_map)

        elif noise_model == 'hetero':

            if task_1 is 'regression':
                loss_func_task_1 = LossFunction_HeteroReg(loss_type=self.multitask_param.loss_1)
            elif task_1 is 'segmentation':
                loss_func_task_1 = LossFunction_HeteroSeg(n_class=self.multitask_param.num_classes[0],
                                                          loss_type=self.multitask_param.loss_1)
            if task_2 is 'regression':
                loss_func_task_2 = LossFunction_HeteroReg(loss_type=self.multitask_param.loss_2)
            elif task_2 is 'segmentation':
                loss_func_task_2 = LossFunction_HeteroSeg(n_class=self.multitask_param.num_classes[1],
                                                          loss_type=self.multitask_param.loss_2)

            data_loss_task_1 = loss_func_task_1(prediction=prediction_task_1,
                                                ground_truth=ground_truth_task_1,
                                                noise=noise_task_1,
                                                weight_map=weight_map)

            data_loss_task_2 = loss_func_task_2(prediction=prediction_task_2,
                                                ground_truth=ground_truth_task_2,
                                                noise=noise_task_2,
                                                weight_map=weight_map)

        elif noise_model == 'single-hetero':

            if task_1 is 'regression':
                loss_func_task_1 = LossFunction_HeteroReg(loss_type=self.multitask_param.loss_1)
            elif task_1 is 'segmentation':
                loss_func_task_1 = LossFunction_HeteroSeg(n_class=self.multitask_param.num_classes[0],
                                                          loss_type=self.multitask_param.loss_1)

            data_loss_task_1 = loss_func_task_1(prediction=prediction_task_1,
                                                ground_truth=ground_truth_task_1,
                                                noise=noise_task_1,
                                                weight_map=weight_map)
            data_loss_task_2 = None

        else:

            data_loss_task_1 = None
            data_loss_task_2 = None

        return data_loss_task_1, data_loss_task_2

    def interpret_output(self, batch_output):
        if not self.is_training:
            return self.output_decoder.decode_batch(
                batch_output['window'], batch_output['location'])
        else:
            return True
