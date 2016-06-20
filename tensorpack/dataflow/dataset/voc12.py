# -*- coding: UTF-8 -*-
# File: voc12.py
# Author: philipcheng
# Time: 6/14/16 -> 9:23 AM
import os
import tarfile
from PIL import Image
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorpack.utils import get_rng, get_dataset_dir
from tensorpack.utils.fs import *
from tensorpack.dataflow.base import DataFlow
from tensorpack.tfutils.symbolic_functions import *

__all__ = ['VOC12SegMeta', 'VOC12Meta', 'VOC12Seg', 'VOC12_ORIGIN_URL']

VOC12_ORIGIN_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'

"""
Preprocessing like this:
    cd train
    for i in *.tar; do dir=${i%.tar}; echo $dir; mkdir -p $dir; tar xf $i -C $dir; done
"""


class VOC12Meta(object):
    """
    Provide base class for metadata
    """
    __metaclass__ = ABCMeta

    def __init__(self, name=None, meta_dir=None):
        if not name:
            name = 'VOC12'
        if not meta_dir:
            meta_dir = get_dataset_dir(name)
        self.dir = meta_dir
        self.VOC12_url = VOC12_ORIGIN_URL
        mkdir_p(self.dir)
        self.classes = self.pascal_classes()
        if self._need_download():
            self._download_meta()

    @staticmethod
    def get_image_path_list(image_dir, image_basename_list, ext):
        image_path_list = [os.path.join(image_dir, image_basename + ext) for image_basename in image_basename_list]
        return image_path_list

    @staticmethod
    def pascal_classes():
        classes = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8,
                   'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                   'potted-plant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tv/monitor': 20}
        return classes

    def _need_download(self):
        """"""
        subdirs = set(get_subdirs(self.dir))
        need_dirs = set(['ImageSets', 'JPEGImages', 'SegmentationClass', 'SegmentationObject'])
        return not subdirs.issuperset(need_dirs)

    def _download_meta(self):
        """"""
        file_path = download(self.VOC12_url, self.dir)
        tarfile.open(file_path, 'r:*').extractall(self.dir)
        ret = os.system('cd {} && mv VOCdevkit/VOC2012/* ./ && rm -fr VOCdevkit'.format(self.dir))
        assert ret == 0, "adjust the files in VOCdevkit wrong"
        if self._need_download():
            raise ValueError('voc12 dataset Error!')

    @staticmethod
    def get_per_channel_mean():
        """
        channel mean for rgb
        :return:
        """
        rgb_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
        return rgb_mean


class VOC12SegMeta(VOC12Meta):
    def __init__(self, meta_dir=None):
        super(VOC12SegMeta, self).__init__(meta_dir=meta_dir)
        self.palette = self.pascal_palette()
        self.jpeg_dir = os.path.join(self.dir, 'JPEGImages')
        self.png_dir = os.path.join(self.dir, 'SegmentationClass')        
        
    def get_image_basename_list(self, dataset_type):
        """"""
        txt_name = os.path.join(self.dir, 'ImageSets', 'Segmentation', dataset_type + '.txt')
        assert os.path.isfile(txt_name)
        basename_list = []
        with open(txt_name) as f:
            for line in f.readlines():
                basename = line.strip()
                basename_list.append(basename)
        return basename_list

    def get_image_label_list(self, dataset_type):
        """
        :param dataset_type: 'train' or 'val' or 'test'
        :returns list of (image,label) pair
        """
        assert dataset_type in ['train', 'val', 'trainval']
        is_training = dataset_type == 'train'
        image_basename_list = self.get_image_basename_list(dataset_type)
        image_path_list = self.get_image_path_list(self.jpeg_dir, image_basename_list, '.jpg')
        if is_training:
            label_path_list = self.get_image_path_list(self.png_dir, image_basename_list, '.png')
        else:
            label_path_list = [None] * len(image_path_list)
        return zip(image_path_list, label_path_list)

    @staticmethod
    def pascal_palette():
        """
        It is for rgb order
        :return:
        """
        rgb_palette = {(0, 0, 0): 0, (128, 0, 0): 1, (0, 128, 0): 2, (128, 128, 0): 3, (0, 0, 128): 4, (128, 0, 128): 5,
                   (0, 128, 128): 6, (128, 128, 128): 7, (64, 0, 0): 8, (192, 0, 0): 9, (64, 128, 0): 10,
                   (192, 128, 0): 11, (64, 0, 128): 12, (192, 0, 128): 13, (64, 128, 128): 14, (192, 128, 128): 15,
                   (0, 64, 0): 16, (128, 64, 0): 17, (0, 192, 0): 18, (128, 192, 0): 19, (0, 64, 128): 20}
        return rgb_palette

    def convert_from_color_segmentation(self, label):
        """
        Convert three channal image labels to one channel image label
        :param label: (h,w,3), bgr order
        :return: (h,w)
        """
        assert len(label.shape) == 3 and label.shape[2] == 3
        label = cast_type(label, 'uint8')
        new_label = np.apply_along_axis(lambda key: self.palette.get(tuple(key[::-1]), 0), axis=2, arr=label)
        return new_label

    def convert_from_class_segmentation(self, label):
        """
        Convert (h,w) label to (h,w,3) colorful image
        :param label: (h,w)
        :return:
        """
        assert type(label) is np.ndarray and len(label.shape) == 2
        new_label = Image.fromarray(label)
        new_label.putpalette(self.palette)
        return np.array(new_label)


class VOC12Seg(DataFlow):
    def __init__(self, dataset_type, meta_dir=None, shuffle=True):
        """
        :param dataset_type: 'train' or 'val' or 'test'
        """
        assert dataset_type in ['train', 'trainval', 'val']
        self.meta = VOC12SegMeta(meta_dir)
        self.num_classes = 21
        self.shuffle = shuffle
        self.img_label_list = self.meta.get_image_label_list(dataset_type)
        self.rng = get_rng(self)

    def size(self):
        return len(self.img_label_list)

    def reset_state(self):
        """
        reset rng for shuffle
        """
        self.rng = get_rng(self)

    def get_data(self):
        """
        Produce original images or shape [h, w, 3], and label
        """
        idxs = np.arange(len(self.img_label_list))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            image_path, label_path = self.img_label_list[k]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR) #bgr
            assert image is not None and image.ndim in (2, 3)
            if image.ndim == 2:
                image = np.expand_dims(image, 2).repeat(3, 2)
            image = image - self.meta.get_per_channel_mean()[::-1]
            if label_path is None:
                label = None
            else:
                label = cv2.imread(label_path, cv2.IMREAD_COLOR)
                assert label is not None and label.ndim in (2, 3)
                if label.ndim == 2:
                    continue
                label = self.meta.convert_from_color_segmentation(label)
            yield [image, label]

if __name__ == '__main__':
    seg = VOC12Seg(dataset_type='train')
    for i in xrange(10):
        data = seg.get_data().next()



