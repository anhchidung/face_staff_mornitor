import cv2
import numpy as np
from skimage import transform as trans

"""
    Face landmarks align
"""

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32) # two eyes, 2 mouth, nose (112X112 pixels)

arcface_src = np.expand_dims(arcface_src, axis=0)


def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')

    assert image_size == 112
    src = arcface_src

    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112):
    M, pose_index = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def align_face(image, landmark):
    if len(landmark) == 0:
        return None
    pts5 = np.array([landmark])
    pts5 = pts5[0, :]
    nimg = norm_crop(image, pts5)
    return nimg
