# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet


class MTHeteroHighRes3DNet(BaseNet):
    # task-dependent networks: 1) regression --> num_classes_task_regression = 1
    #                             task_1
    #                          2) segmentation --> num_classes_task_segmentation = k
    #                             task_2
    #
    #
    # Epistemic + heteroscedatic modelling
    #
    #               2 networks for each tasks
    #               e.g.                        --- task 1 mean --->
    #                                           --- task 1 var  --->
    #                    representation network                      ---> combined loss
    #                                           --- task 2 mean --->
    #                                           --- task 2 var  --->
    def __init__(self,
                 num_classes,
                 dropout_rep,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='MTHeteroHighRes3DNet'):

        super(MTHeteroHighRes3DNet, self).__init__(
            num_classes=num_classes,
            dropout_rep=dropout_rep,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        # representation layer: 0, 1, 2, 3, 4
        # task 1 layer (pred): 5, 6
        # task 1 layer (noise): 7, 8
        # task 2 layer (pred): 9, 10
        # task 2 layer (noise): 11, 12
        # task 1 pred out: 13
        # task 1 noise out: 14
        # task 2 pred out: 15
        # task 2 noise out: 16

        self.layers = [
            {'name': 'conv_0', 'n_features': 16, 'kernel_size': 3},
            {'name': 'res_1', 'n_features': 16, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'res_2', 'n_features': 32, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'res_3', 'n_features': 64, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'conv_1', 'n_features': 80, 'kernel_size': 1},

            {'name': 'task_1_mean_fc_1', 'n_features': 80, 'kernel_size': 1},
            {'name': 'task_1_mean_fc_2', 'n_features': 80, 'kernel_size': 1},
            {'name': 'task_1_noise_fc_1', 'n_features': 80, 'kernel_size': 1},
            {'name': 'task_1_noise_fc_2', 'n_features': 80, 'kernel_size': 1},

            {'name': 'task_2_mean_fc_1', 'n_features': 80, 'kernel_size': 1},
            {'name': 'task_2_mean_fc_2', 'n_features': 80, 'kernel_size': 1},
            {'name': 'task_2_noise_fc_1', 'n_features': 80, 'kernel_size': 1},
            {'name': 'task_2_noise_fc_2', 'n_features': 80, 'kernel_size': 1},

            {'name': 'task_1_mean_fc_out', 'n_features': num_classes[0], 'kernel_size': 1},
            {'name': 'task_1_noise_fc_out', 'n_features': 1, 'kernel_size': 1},

            {'name': 'task_2_mean_fc_out', 'n_features': num_classes[1], 'kernel_size': 1},
            {'name': 'task_2_noise_fc_out', 'n_features': 1, 'kernel_size': 1}]

    def layer_op(self, images, is_training, layer_id=-1):
        assert layer_util.check_spatial_dims(
            images, lambda x: x % 8 == 0)
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []

        ###### representation network ######

        ### first convolution layer
        params = self.layers[0]
        first_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = first_conv_layer(images, is_training)
        layer_instances.append((first_conv_layer, flow))

        ### resblocks, all kernels dilated by 1 (normal convolution)
        params = self.layers[1]
        with DilatedTensor(flow, dilation_factor=1) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### resblocks, all kernels dilated by 2
        params = self.layers[2]
        with DilatedTensor(flow, dilation_factor=2) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### resblocks, all kernels dilated by 4
        params = self.layers[3]
        with DilatedTensor(flow, dilation_factor=4) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### 1x1x1 convolution layer
        params = self.layers[4]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fc_layer(flow, is_training, keep_prob=self.dropout_rep)
        layer_instances.append((fc_layer, flow))

        ###### TASK 1 - prediction ######

        ### 1x1x1 convolution layer for task_1
        params = self.layers[5]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task1_mean_1 = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow_task1_mean_1))

        ### 1x1x1 convolution layer for task_1
        params = self.layers[6]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task1_mean_2 = fc_layer(flow_task1_mean_1, is_training)
        layer_instances.append((fc_layer, flow_task1_mean_2))

        ###### TASK 1 - prediction ######

        params = self.layers[7]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task1_noise_1 = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow_task1_noise_1))

        ### 1x1x1 convolution layer for task_1
        params = self.layers[8]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task1_noise_2 = fc_layer(flow_task1_noise_1, is_training)
        layer_instances.append((fc_layer, flow_task1_noise_2))

        ###### TASK 2 - prediction ######

        ### 1x1x1 convolution layer for task_2
        params = self.layers[9]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task2_mean_1 = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow_task2_mean_1))

        ### 1x1x1 convolution layer for task_2
        params = self.layers[10]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task2_mean_2 = fc_layer(flow_task2_mean_1, is_training)
        layer_instances.append((fc_layer, flow_task2_mean_2))

        ###### TASK 2 - noise ######

        params = self.layers[11]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task2_noise_1 = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow_task2_noise_1))

        ### 1x1x1 convolution layer for task_1
        params = self.layers[12]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task2_noise_2 = fc_layer(flow_task2_noise_1, is_training)
        layer_instances.append((fc_layer, flow_task2_noise_2))

        ###### OUTPUT TASK 1 - mean ######

        ### 1x1x1 convolution layer output for task_1
        params = self.layers[13]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task1_mean_output = fc_layer(flow_task1_mean_2, is_training)
        layer_instances.append((fc_layer, flow_task1_mean_output))

        ###### OUTPUT TASK 1 - noise ######

        ### 1x1x1 convolution layer for task_1
        params = self.layers[14]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task1_noise_output = fc_layer(flow_task1_noise_2, is_training)
        layer_instances.append((fc_layer, flow_task1_noise_output))

        ###### OUTPUT TASK 2 - mean ######

        ### 1x1x1 convolution layer output for task_2
        params = self.layers[15]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task2_mean_output = fc_layer(flow_task2_mean_2, is_training)
        layer_instances.append((fc_layer, flow_task2_mean_output))

        ###### OUTPUT TASK 2 - noise ######

        ### 1x1x1 convolution layer for task_2
        params = self.layers[16]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow_task2_noise_output = fc_layer(flow_task2_noise_2, is_training)
        layer_instances.append((fc_layer, flow_task2_noise_output))

        # set training properties
        if is_training:
            self._print(layer_instances)
            final_task_tensors = [item[1] for item in layer_instances]
            return final_task_tensors[-4:]
        return [item[1] for item in layer_instances[-4:]]

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)


class HighResBlock(TrainableLayer):
    """
    This class define a high-resolution block with residual connections
    kernels

        - specify kernel sizes of each convolutional layer
        - e.g.: kernels=(5, 5, 5) indicate three conv layers of kernel_size 5

    with_res

        - whether to add residual connections to bypass the conv layers
    """

    def __init__(self,
                 n_output_chns,
                 kernels=(3, 3),
                 acti_func='relu',
                 w_initializer=None,
                 w_regularizer=None,
                 with_res=True,
                 name='HighResBlock'):

        super(HighResBlock, self).__init__(name=name)

        self.n_output_chns = n_output_chns
        if hasattr(kernels, "__iter__"):  # a list of layer kernel_sizes
            self.kernels = kernels
        else:  # is a single number (indicating single layer)
            self.kernels = [kernels]
        self.acti_func = acti_func
        self.with_res = with_res

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training):
        output_tensor = input_tensor
        for (i, k) in enumerate(self.kernels):
            # create parameterised layers
            bn_op = BNLayer(regularizer=self.regularizers['w'],
                            name='bn_{}'.format(i))
            acti_op = ActiLayer(func=self.acti_func,
                                regularizer=self.regularizers['w'],
                                name='acti_{}'.format(i))
            conv_op = ConvLayer(n_output_chns=self.n_output_chns,
                                kernel_size=k,
                                stride=1,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                name='conv_{}'.format(i))
            # connect layers
            output_tensor = bn_op(output_tensor, is_training)
            output_tensor = acti_op(output_tensor)
            output_tensor = conv_op(output_tensor)
        # make residual connections
        if self.with_res:
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)
        return output_tensor