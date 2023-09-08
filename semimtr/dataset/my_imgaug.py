import cv2
import numpy as np
import random
import imgaug.augmenters as iaa
import imgaug as ia

class RotateAffine:
    def __init__(self, angle, p=0.5):
        self._angle = angle
        self._p = p
    
    def warpAffine(self, src, M, dsize, from_bounding_box_only=False):
        color = tuple(src[0:1, 0:1, 0:][0][0])
        color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
        src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        image = cv2.warpAffine(src, M, dsize, borderValue=color)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def Rotate(self, image, bias=0):
        # get dims, find center
        v, max_v = self._angle
        angle = random.randint(v, max_v)
        image = np.array(image)
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        image = self.warpAffine(image, M, (nW, nH), False)
        return image
    
    def func_images(self, images, random_state, parents, hooks):
        result = []
        for image in images:
            if random.random() >= self._p:
                image_aug = self.Rotate(image)
            else:
                image_aug = image
            result.append(image_aug)
        result = np.array(result)
        return result

    def __call__(self):
        return iaa.Lambda(func_images=self.func_images)