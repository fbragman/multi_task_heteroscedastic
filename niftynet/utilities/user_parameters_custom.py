# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from niftynet.utilities.user_parameters_helper import add_input_name_args
from niftynet.utilities.user_parameters_helper import int_array
from niftynet.utilities.user_parameters_helper import str2boolean


#######################################################################
# To support a CUSTOM_SECTION in config file:
# (e.g., MYTASK; in parallel with SEGMENTATION, REGRESSION etc.)
#
# 1) update niftynet.utilities.user_parameters_custom.SUPPORTED_ARG_SECTIONS
# with a key-value pair:
# where the key should be MYTASK, a standardised string --
# Standardised string is defined in
# niftynet.utilities.user_parameters_helper.standardise_string
# the section name will be filtered with,
# re.sub('[^0-9a-zA-Z_\- ]+', '', input_string.strip())
#
# the value should be __add_mytask_args()
#
# 2) create a function __add_mytask_args() with task specific arguments
# this function should return an argparse obj
#
# 3) in the application file, specify:
# `REQUIRED_CONFIG_SECTION = "MYTASK"`
# so that the application will have access to the task specific arguments
#########################################################################


def add_customised_args(parser, task_name):
    task_name = task_name.upper()
    if task_name in SUPPORTED_ARG_SECTIONS:
        return SUPPORTED_ARG_SECTIONS[task_name](parser)
    else:
        raise NotImplementedError


def __add_regression_args(parser):
    parser.add_argument(
        "--loss_border",
        metavar='',
        help="Set the border size for the loss function to ignore",
        type=int,
        default=0)

    parser.add_argument(
        "--error_map",
        metavar='',
        help="Set whether to output the regression error maps (the maps "
             "will be stored in $model_dir/error_maps; the error maps "
             "can be used for window sampling).",
        type=str2boolean,
        default=False)

    from niftynet.application.regression_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_multitask_args(parser):
    parser.add_argument(
        "--loss_border",
        metavar='',
        help="Set the border size for the loss function to ignore",
        type=int,
        default=0)

    parser.add_argument(
        "--noise_model",
        metavar='TYPE_STR',
        help="Homoscedatic or heteroscedatic noise modelling",
        default='homo')

    parser.add_argument(
        "--num_classes",
        help="Set number of classes for each task",
        type=int_array,
        default=-1)

    parser.add_argument(
        "--loss_1",
        metavar='TYPE_STR',
        help="[Training only] loss function for task 1 type_str",
        default='CrossEntropy')

    parser.add_argument(
        "--loss_2",
        metavar='TYPE_STR',
        help="[Training only] loss function for task 2 type_str",
        default='L2Loss')

    parser.add_argument(
        "--multitask_loss",
        metavar='TYPE_STR',
        help="[Training only] type of loss function for multi-task application",
        default='average')

    parser.add_argument(
        "--output_prob",
        metavar='',
        help="[Inference only] whether to output multi-class probabilities",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--label_normalisation",
        metavar='',
        help="whether to map unique labels in the training set to "
             "consecutive integers (the smallest label will be  mapped to 0)",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--loss_sigma_1",
        help='Initial value for task 1 homoscedatic noise or multi-task weight',
        type=float,
        default=1)

    parser.add_argument(
        "--loss_sigma_2",
        help='Initial value for task 2 homoscedatic noise or multi-task weight',
        type=float,
        default=1)

    parser.add_argument(
        "--output_interp_order_task1",
        help='Output for task 1',
        type=int,
        default=3)

    parser.add_argument(
        "--output_interp_order_task2",
        help='Output for task 2',
        type=int,
        default=0)

    parser.add_argument(
        "--seg_T_passes",
        help="Stochastic T passes to calculate expected log-likelihood for segmentation",
        type=int,
        default=10)

    parser.add_argument(
        "--dropout_representation",
        help="Drop out probability on final layer of representation network",
        type=float,
        default=0)

    from niftynet.application.multitask_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_segmentation_args(parser):
    parser.add_argument(
        "--num_classes",
        metavar='',
        help="Set number of classes",
        type=int,
        default=-1)

    parser.add_argument(
        "--output_prob",
        metavar='',
        help="[Inference only] whether to output multi-class probabilities",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--label_normalisation",
        metavar='',
        help="whether to map unique labels in the training set to "
             "consecutive integers (the smallest label will be  mapped to 0)",
        type=str2boolean,
        default=False)

    # for selective sampling only
    parser.add_argument(
        "--min_sampling_ratio",
        help="[Training only] Minimum ratio of samples in a window for "
             "selective sampler",
        metavar='',
        type=float,
        default=0
    )

    # for selective sampling only
    parser.add_argument(
        "--compulsory_labels",
        help="[Training only] List of labels to have in the window for "
             "selective sampling",
        metavar='',
        type=int_array,
        default=(0, 1)
    )

    # for selective sampling only
    parser.add_argument(
        "--rand_samples",
        help="[Training only] Number of completely random samples per image "
             "when using selective sampler",
        metavar='',
        type=int,
        default=0
    )

    # for selective sampling only
    parser.add_argument(
        "--min_numb_labels",
        help="[Training only] Number of labels to have in the window for "
             "selective sampler",
        metavar='',
        type=int,
        default=1
    )

    # for selective sampling only
    parser.add_argument(
        "--proba_connect",
        help="[Training only] Number of labels to have in the window for "
             "selective sampler",
        metavar='',
        type=str2boolean,
        default=True
    )

    parser.add_argument(
        "--evaluation_units",
        help="Compute per-component metrics for per label or per connected "
             "component. [foreground, label, or cc]",
        choices = ['foreground', 'label', 'cc'],
        default='foreground')

    from niftynet.application.segmentation_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_gan_args(parser):
    parser.add_argument(
        "--noise_size",
        metavar='',
        help="length of the noise vector",
        type=int,
        default=-1)

    parser.add_argument(
        "--n_interpolations",
        metavar='',
        help="the method of generating window from image",
        type=int,
        default=10)

    from niftynet.application.gan_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_classification_args(parser):
    parser.add_argument(
        "--num_classes",
        metavar='',
        help="Set number of classes",
        type=int,
        default=-1)

    parser.add_argument(
        "--output_prob",
        metavar='',
        help="[Inference only] whether to output multi-class probabilities",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--label_normalisation",
        metavar='',
        help="whether to map unique labels in the training set to "
             "consecutive integers (the smallest label will be  mapped to 0)",
        type=str2boolean,
        default=False)


    from niftynet.application.classification_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_autoencoder_args(parser):
    from niftynet.application.autoencoder_application import SUPPORTED_INFERENCE
    parser.add_argument(
        "--inference_type",
        metavar='',
        help="choose an inference type_str for the trained autoencoder",
        choices=list(SUPPORTED_INFERENCE))

    parser.add_argument(
        "--noise_stddev",
        metavar='',
        help="standard deviation of noise when inference type_str is sample",
        type=float)

    parser.add_argument(
        "--n_interpolations",
        metavar='',
        help="the method of generating window from image",
        type=int,
        default=10)

    from niftynet.application.autoencoder_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_registration_args(parser):
    parser.add_argument(
        "--label_normalisation",
        metavar='',
        help="whether to map unique labels in the training set to "
             "consecutive integers (the smallest label will be  mapped to 0)",
        type=str2boolean,
        default=False)

    from niftynet.application.label_driven_registration import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


SUPPORTED_ARG_SECTIONS = {
    'REGRESSION': __add_regression_args,
    'SEGMENTATION': __add_segmentation_args,
    'CLASSIFICATION': __add_classification_args,
    'AUTOENCODER': __add_autoencoder_args,
    'GAN': __add_gan_args,
    'MULTITASK': __add_multitask_args,
    'REGISTRATION': __add_registration_args
}
