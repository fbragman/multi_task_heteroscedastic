# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.utilities.util_common import ParserNamespace

# test multiple modalities
MULTI_MOD_DATA = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1reader.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None
    ),
    'FLAIR': ParserNamespace(
        csv_file=os.path.join('testing_data', 'FLAIRreader.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None
    )
}
MULTI_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'))

# test single modalities
SINGLE_MOD_DATA = {
    'lesion': ParserNamespace(
        csv_file=os.path.join('testing_data', 'lesion.csv'),
        path_to_search='testing_data',
        filename_contains=('Lesion',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None
    )
}
SINGLE_MOD_TASK = ParserNamespace(image=('lesion',))

EXISTING_DATA = {
    'lesion': ParserNamespace(
        csv_file=os.path.join('testing_data', 'lesion.csv'),
        interp_order=3,
        pixdim=None,
        axcodes=None
    )
}

# test labels
LABEL_DATA = {
    'parcellation': ParserNamespace(
        csv_file=os.path.join('testing_data', 'labels.csv'),
        path_to_search='testing_data',
        filename_contains=('Parcellation',),
        filename_not_contains=('Lesion',),
        interp_order=0,
        pixdim=(3, 3.9, 3),
        axcodes=None
    )
}
LABEL_TASK = ParserNamespace(label=('parcellation',))

BAD_DATA = {
    'lesion': ParserNamespace(
        csv_file=os.path.join('testing_data', 'lesion.csv'),
        path_to_search='testing_data',
        filename_contains=('Lesion',),
        filename_not_contains=('Parcellation',),
        pixdim=None,
        axcodes=None
    )
}
BAD_TASK = ParserNamespace(image=('test',))

# default data_partitioner
data_partitioner = ImageSetsPartitioner()
multi_mod_list = data_partitioner.initialise(MULTI_MOD_DATA).get_file_list()
single_mod_list = data_partitioner.initialise(SINGLE_MOD_DATA).get_file_list()
existing_list = data_partitioner.initialise(EXISTING_DATA).get_file_list()
label_list = data_partitioner.initialise(LABEL_DATA).get_file_list()
bad_data_list = data_partitioner.initialise(BAD_DATA).get_file_list()


class ImageReaderTest(tf.test.TestCase):
    def test_initialisation(self):
        with self.assertRaisesRegexp(ValueError, ''):
            reader = ImageReader(['test'])
            reader.initialise(MULTI_MOD_DATA, MULTI_MOD_TASK, multi_mod_list)
        with self.assertRaisesRegexp(AssertionError, ''):
            reader = ImageReader(None)
            # reader.initialise(MULTI_MOD_DATA, MULTI_MOD_TASK, multi_mod_list)

        reader = ImageReader(['image'])
        reader.initialise(MULTI_MOD_DATA, MULTI_MOD_TASK, multi_mod_list)
        self.assertEqual(len(reader.output_list), 4)

        reader = ImageReader(['image'])
        reader.initialise(SINGLE_MOD_DATA, SINGLE_MOD_TASK, single_mod_list)
        self.assertEqual(len(reader.output_list), 4)

        reader = ImageReader(['image'])
        with self.assertRaisesRegexp(ValueError, ''):
            reader.initialise(SINGLE_MOD_DATA, SINGLE_MOD_TASK, [])

    def test_properties(self):
        reader = ImageReader(['image'])
        reader.initialise(SINGLE_MOD_DATA, SINGLE_MOD_TASK, single_mod_list)
        self.assertEqual(len(reader.output_list), 4)
        self.assertDictEqual(reader.shapes,
                             {'image': (256, 168, 256, 1, 1)})
        self.assertDictEqual(reader.tf_dtypes, {'image': tf.float32})
        self.assertEqual(reader.names, ['image'])
        self.assertDictEqual(reader.input_sources,
                             {'image': ('lesion',)})
        self.assertEqual(reader.get_subject_id(1)[:4], 'Fin_')

    def test_existing_csv(self):
        reader_for_csv = ImageReader(['image'])
        reader_for_csv.initialise(
            SINGLE_MOD_DATA, SINGLE_MOD_TASK, single_mod_list)
        reader = ImageReader(['image'])
        reader.initialise(EXISTING_DATA, SINGLE_MOD_TASK, existing_list)
        self.assertEqual(len(reader.output_list), 4)
        self.assertDictEqual(reader.shapes,
                             {'image': (256, 168, 256, 1, 1)})
        self.assertDictEqual(reader.tf_dtypes, {'image': tf.float32})
        self.assertEqual(reader.names, ['image'])
        self.assertDictEqual(reader.input_sources,
                             {'image': ('lesion',)})
        self.assertEqual(reader.get_subject_id(1)[:4], 'Fin_')

    def test_operations(self):
        reader = ImageReader(['image'])
        reader.initialise(SINGLE_MOD_DATA, SINGLE_MOD_TASK, single_mod_list)
        idx, data, interp_order = reader()
        self.assertEqual(
            SINGLE_MOD_DATA['lesion'].interp_order, interp_order['image'][0])
        self.assertAllClose(data['image'].shape, (256, 168, 256, 1, 1))

    def test_preprocessing(self):
        reader = ImageReader(['image'])
        reader.initialise(SINGLE_MOD_DATA, SINGLE_MOD_TASK, single_mod_list)
        idx, data, interp_order = reader()
        self.assertEqual(SINGLE_MOD_DATA['lesion'].interp_order,
                         interp_order['image'][0])
        self.assertAllClose(data['image'].shape, (256, 168, 256, 1, 1))
        reader.add_preprocessing_layers(
            [PadLayer(image_name=['image'], border=(10, 5, 5))])
        idx, data, interp_order = reader(idx=2)
        self.assertEqual(idx, 2)
        self.assertAllClose(data['image'].shape, (276, 178, 266, 1, 1))

    def test_preprocessing_zero_padding(self):
        reader = ImageReader(['image'])
        reader.initialise(SINGLE_MOD_DATA, SINGLE_MOD_TASK, single_mod_list)
        idx, data, interp_order = reader()
        self.assertEqual(SINGLE_MOD_DATA['lesion'].interp_order,
                         interp_order['image'][0])
        self.assertAllClose(data['image'].shape, (256, 168, 256, 1, 1))
        reader.add_preprocessing_layers(
            [PadLayer(image_name=['image'], border=(0, 0, 0))])
        idx, data, interp_order = reader(idx=2)
        self.assertEqual(idx, 2)
        self.assertAllClose(data['image'].shape, (256, 168, 256, 1, 1))

    def test_trainable_preprocessing(self):
        label_file = os.path.join('testing_data', 'label_reader.txt')
        if os.path.exists(label_file):
            os.remove(label_file)
        label_normaliser = DiscreteLabelNormalisationLayer(
            image_name='label',
            modalities=vars(LABEL_TASK).get('label'),
            model_filename=os.path.join('testing_data', 'label_reader.txt'))
        reader = ImageReader(['label'])
        with self.assertRaisesRegexp(AssertionError, ''):
            reader.add_preprocessing_layers(label_normaliser)
        reader.initialise(LABEL_DATA, LABEL_TASK, label_list)
        reader.add_preprocessing_layers(label_normaliser)
        reader.add_preprocessing_layers(
            [PadLayer(image_name=['label'], border=(10, 5, 5))])
        idx, data, interp_order = reader(idx=0)
        unique_data = np.unique(data['label'])
        expected_v1 = np.array(
            [0., 1., 2., 3., 4., 5., 6., 7., 8.,
             9., 10., 11., 12., 13., 14., 15., 16., 17.,
             18., 19., 20., 21., 22., 23., 24., 25., 26., 27.,
             28., 29., 30., 31., 32., 33., 34., 35., 36.,
             37., 38., 39., 40., 41., 42., 43., 44., 45.,
             46., 47., 48., 49., 50., 51., 52., 53., 54.,
             55., 56., 57., 58., 59., 60., 61., 62., 63.,
             64., 65., 66., 67., 68., 69., 70., 71., 72.,
             73., 74., 75., 76., 77., 78., 79., 80., 81.,
             82., 83., 84., 85., 86., 87., 88., 89., 90.,
             91., 92., 93., 94., 95., 96., 97., 98., 99.,
             100., 101., 102., 103., 104., 105., 106., 107., 108.,
             109., 110., 111., 112., 113., 114., 115., 116., 117.,
             118., 119., 120., 121., 122., 123., 124., 125., 126.,
             127., 128., 129., 130., 131., 132., 133., 134., 135.,
             136., 137., 138., 139., 140., 141., 142., 143., 144.,
             145., 146., 147., 148., 149., 150., 151., 152., 153.,
             154., 155., 156., 157.], dtype=np.float32)
        expected_v2 = np.array(
            [0., 1., 2., 3., 4., 5., 6., 7., 8.,
             9., 10., 11., 12., 13., 14., 15., 16., 17.,
             18., 20., 21., 22., 23., 24., 25., 26., 27.,
             28., 29., 30., 31., 32., 33., 34., 35., 36.,
             37., 38., 39., 40., 41., 42., 43., 44., 45.,
             46., 47., 48., 49., 50., 51., 52., 53., 54.,
             55., 56., 57., 58., 59., 60., 61., 62., 63.,
             64., 65., 66., 67., 68., 69., 70., 71., 72.,
             73., 74., 75., 76., 77., 78., 79., 80., 81.,
             82., 83., 84., 85., 86., 87., 88., 89., 90.,
             91., 92., 93., 94., 95., 96., 97., 98., 99.,
             100., 101., 102., 103., 104., 105., 106., 107., 108.,
             109., 110., 111., 112., 113., 114., 115., 116., 117.,
             118., 119., 120., 121., 122., 123., 124., 125., 126.,
             127., 128., 129., 130., 131., 132., 133., 134., 135.,
             136., 137., 138., 139., 140., 141., 142., 143., 144.,
             145., 146., 147., 148., 149., 150., 151., 152., 153.,
             154., 155., 156., 157.], dtype=np.float32)
        compatible_assert = \
            np.all(unique_data == expected_v1) or \
            np.all(unique_data == expected_v2)
        self.assertTrue(compatible_assert)
        self.assertAllClose(data['label'].shape, (103, 74, 93, 1, 1))

    def test_errors(self):
        with self.assertRaisesRegexp(AttributeError, ''):
            reader = ImageReader(['image'])
            reader.initialise(BAD_DATA, SINGLE_MOD_TASK, bad_data_list)
        with self.assertRaisesRegexp(ValueError, ''):
            reader = ImageReader(['image'])
            reader.initialise(SINGLE_MOD_DATA, BAD_TASK, single_mod_list)

        reader = ImageReader(['image'])
        reader.initialise(SINGLE_MOD_DATA, SINGLE_MOD_TASK, single_mod_list)
        idx, data, interp_order = reader(idx=100)
        self.assertEqual(idx, -1)
        self.assertEqual(data, None)
        idx, data, interp_order = reader(shuffle=True)
        self.assertEqual(data['image'].shape, (256, 168, 256, 1, 1))


if __name__ == "__main__":
    tf.test.main()
