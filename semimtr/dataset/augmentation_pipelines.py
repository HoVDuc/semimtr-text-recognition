import cv2
import random
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from semimtr.dataset.my_imgaug import RotateAffine


def get_augmentation_pipeline(augmentation_severity=1):
    """
    Defining the augmentation pipeline for SemiMTR pre-training and fine-tuning.
    :param augmentation_severity:
        0 - ABINet augmentation pipeline
        1 - SemiMTR augmentation pipeline
        2 - SeqCLR augmentation pipeline
    :return: augmentation_pipeline
    """
    if augmentation_severity == 1:
        augmentations = iaa.Sequential([
            # iaa.Invert(0.5), # Đảo ngược màu sắc 
            RotateAffine((-20, 20), p=0.5)(),
            iaa.OneOf([
                # iaa.ChannelShuffle(0.35), # Hoán đổi các kênh màu
                iaa.Grayscale(alpha=(0.0, 1.0)), # Chuyển đổi hình ảnh xám
                iaa.KMeansColorQuantization(), # Áp dụng phân đoạn màu sắc
                iaa.HistogramEqualization(), # Cân bằng histogram
                iaa.Dropout(p=(0, 0.2), per_channel=0.5), # Loại bỏ một phần của hình ảnh 
                iaa.GammaContrast((0.5, 2.0)), # Tăng cường độ tương phản
                iaa.MultiplyBrightness((0.5, 1.5)), # Tăng hoặc giảm độ sáng
                # iaa.AddToHueAndSaturation((-50, 50), per_channel=True), # Thay đổi màu sắc và độ bão hòa
                # iaa.ChangeColorTemperature((1100, 10000)) # Thay đổi nhiệt độ màu 
            ]),
            iaa.OneOf([
                iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)), # Làm sắc nét hình ảnh
                iaa.OneOf([
                    iaa.GaussianBlur((0.5, 1.5)), # Làm mờ hình ảnh bằng bộ lọc Gaussian
                    iaa.AverageBlur(k=(2, 6)), # Làm mờ hình ảnh bằng bộ lọc trung bình
                    iaa.MedianBlur(k=(3, 7)), # Làm mờ hình ảnh bằng bộ lọc trung vị
                    iaa.MotionBlur(k=5) # Làm mờ hình ảnh bằng bộ lọc chuyển động
                ])
            ]),
            iaa.OneOf([
                iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)), # Thêm nhiễu Gaussian vào hình ảnh 
                iaa.ImpulseNoise(0.1), # Thêm nhiễu tương tự xung
                iaa.MultiplyElementwise((0.5, 1.5)) # Nhân từng pixel của hình ảnh với một hệ số 
            ]),
        ])
    elif augmentation_severity == 2:
        optional_augmentations_list = [
            iaa.LinearContrast((0.5, 1.0)), # Thay đổi độ tương phản tuyến tính
            iaa.GaussianBlur((0.5, 1.5)),
            iaa.Crop(percent=((0, 0.4), (0, 0), (0, 0.4), (0, 0.0)), keep_size=True), # Cắt ảnh theo phần trăm được chỉ định 
            iaa.Crop(percent=((0, 0.0), (0, 0.02), (0, 0), (0, 0.02)), keep_size=True),
            iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)), # Tăng độ sắc nét của hình ảnh bằng cách áp dụng bộ lọc sắc né
            # iaa.PiecewiseAffine(scale=(0.02, 0.03), mode='edge'), # In SeqCLR but replaced with a faster aug
            iaa.ElasticTransformation(alpha=(0, 0.8), sigma=0.25), # Áp dụng biến đổi co dãn linh hoạt để làm biến dạng hình ảnh.
            iaa.PerspectiveTransform(scale=(0.01, 0.02)), # Áp dụng biến đổi góc nhìn để làm biến dạng hình ảnh theo góc độ khác nhau. 
        ]
        augmentations = iaa.SomeOf((1, None), optional_augmentations_list, random_order=True)
    else:
        raise NotImplementedError(f'augmentation_severity={augmentation_severity} is not supported')

    return augmentations
