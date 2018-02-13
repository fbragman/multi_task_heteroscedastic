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

SUPPORTED_INPUT = set(['image', 'output_1', 'output_2', 'weight'])


class MultiTaskApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = 'MULTITASK'

    # Initialise the class/application
    def __init__(self, net_param, action_param, is_training):
        BaseApplication.__init__(self)
        tf.logging.info('starting multi-task application')

        self.is_training = is_training

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

        normalisation_layers = []
        if self.net_param.normalisation:
            normalisation_layers.append(histogram_normaliser)
        if self.net_param.whitening:
            normalisation_layers.append(mean_var_normaliser)

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
            elif self.multitask_param.noise_model == 'hetero':
                net_out_task_1 = net_out[0]
                noise_out_task_1 = net_out[1]
                net_out_task_2 = net_out[2]
                noise_out_task_2 = net_out[3]

            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)

            crop_layer = CropLayer(
                border=self.multitask_param.loss_border, name='crop-88')

            prediction_task_1 = crop_layer(net_out_task_1)
            prediction_task_2 = crop_layer(net_out_task_2)

            ground_truth_task_1 = crop_layer(data_dict.get('output_1', None))
            ground_truth_task_2 = crop_layer(data_dict.get('output_2', None))

            # Make sure ground truth is int32/int64
            ground_truth_task_2 = tf.to_int64(ground_truth_task_2)

            weight_map = None if data_dict.get('weight', None) is None \
                else crop_layer(data_dict.get('weight', None))

            # Set up the noise model
            if self.multitask_param.noise_model == 'homo':

                loss_func_task_1 = LossFunction_Reg(loss_type=self.multitask_param.loss_1)
                loss_func_task_2 = LossFunction_Seg(n_class=self.multitask_param.num_classes[1],
                                                    loss_type=self.multitask_param.loss_2)

                data_loss_task_1 = loss_func_task_1(prediction=prediction_task_1,
                                                    ground_truth=ground_truth_task_1,
                                                    weight_map=weight_map)

                data_loss_task_2 = loss_func_task_2(prediction=prediction_task_2,
                                                    ground_truth=ground_truth_task_2,
                                                    weight_map=weight_map)

            elif self.multitask_param.noise_model == 'hetero':

                loss_func_task_1 = LossFunction_HeteroReg(loss_type=self.multitask_param.loss_1)
                loss_func_task_2 = LossFunction_HeteroSeg(n_class=self.multitask_param.num_classes[1],
                                                          loss_type=self.multitask_param.loss_2)

                if self.multitask_param.hetero_task_init == 'zeros':

                    noise_init_1 = tf.zeros_initializer()
                    noise_init_2 = tf.zeros_initializer()

                elif self.multitask_param.hetero_task_init == 'random':

                    noise_init_1 = tf.truncated_normal_initializer(0, 1)
                    noise_init_2 = tf.truncated_normal_initializer(0, 1)

                elif self.multitask_param.hetero_task_init == 'value':

                    noise_init_1 = tf.constant_initializer(self.multitask_param.hetero_task_1_init)
                    noise_init_2 = tf.constant_initializer(self.multitask_param.hetero_task_2_init)


                noise_task_1 = tf.get_variable('noise_1_img',
                                                shape=tf.shape(prediction_task_1),
                                                dtype=tf.float32,
                                                initializer=noise_init_1,
                                                trainable=True)

                noise_task_2 = tf.get_variable('noise_2_img',
                                               shape=tf.shape(prediction_task_1),
                                               dtype=tf.float32,
                                               initializer=noise_init_2,
                                               trainable=True)


                data_loss_task_1 = loss_func_task_1(prediction=prediction_task_1,
                                                    ground_truth=ground_truth_task_1,
                                                    noise=noise_task_1,
                                                    weight_map=weight_map)

                data_loss_task_2 = loss_func_task_2(prediction=prediction_task_2,
                                                    ground_truth=ground_truth_task_2,
                                                    noise=noise_task_2,
                                                    weight_map=weight_map)

            # Set up the multi-task model
            # Note: if using hetero - only summed_loss should be really used
            #       homosecedatic_1 should only be used with noise_model = 'homo'
            multitask_loss = self.multitask_param.multitask_loss
            mt_loss_task = LossFunction_MT(multitask_loss)

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

            # collecting output variables
            outputs_collector.add_to_collection(
                var=tf.convert_to_tensor(w_1), name='w_1',
                average_over_devices=False, collection=CONSOLE)

            # collecting output variables
            outputs_collector.add_to_collection(
                var=tf.convert_to_tensor(w_2), name='w_2',
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=data_loss, name='Loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

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
            reg_out = net_out[0]
            seg_out = net_out[1]

            # Segmentation
            output_prob = self.multitask_param.output_prob
            num_classes_seg = self.multitask_param.num_classes[1]
            if output_prob and num_classes_seg > 1:
                post_process_layer = PostProcessingLayer(
                    'SOFTMAX', num_classes=num_classes_seg)
            elif not output_prob and num_classes_seg > 1:
                post_process_layer = PostProcessingLayer(
                    'ARGMAX', num_classes=num_classes_seg)
            else:
                post_process_layer = PostProcessingLayer(
                    'IDENTITY', num_classes=num_classes_seg)
            seg_out = post_process_layer(seg_out)

            crop_layer = CropLayer(border=0, name='crop-88')
            post_process_layer = PostProcessingLayer('IDENTITY')
            reg_out = post_process_layer(crop_layer(reg_out))

            # Concatenation of tasks
            if output_prob and num_classes_seg > 1:
                # iterate over the classes and stack them all
                class_imgs = []
                for idx in range(num_classes_seg):
                    class_imgs.append(tf.expand_dims(seg_out[:, :, :, idx], -1))

                for idx in range(num_classes_seg):
                    if idx == 0:
                        mt_out = tf.concat([reg_out, class_imgs[idx]], 3)
                    else:
                        mt_out = tf.concat([mt_out, class_imgs[idx]], 3)
            else:
                mt_out = tf.stack([reg_out, tf.cast(seg_out, tf.float32)], axis=4)

            outputs_collector.add_to_collection(
                var=mt_out, name='window',
                average_over_devices=False, collection=NETWORK_OUTPUT)

            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)

            init_aggregator = \
                self.SUPPORTED_SAMPLING[self.net_param.window_sampling][2]
            init_aggregator()

    def interpret_output(self, batch_output):
        if not self.is_training:
            return self.output_decoder.decode_batch(
                batch_output['window'], batch_output['location'])
        else:
            return True
