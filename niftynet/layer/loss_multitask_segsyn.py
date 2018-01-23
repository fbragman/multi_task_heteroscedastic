# -*- coding: utf-8 -*-
"""
Loss functions for multi-task regression and segmentation
"""
from __future__ import absolute_import, print_function, division

import tensorflow as tf

from niftynet.engine.application_factory import LossRegressionFactory
from niftynet.layer.base_layer import Layer