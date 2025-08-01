import ncnn
import numpy as np
from app_face_core.align_face import align_face


class ArcFaceFeature():
    def __init__(self, param_path, bin_path):
        self.mean_vals = [127.5, 127.5, 127.5]
        self.norm_vals = [1. / 127.5, 1. / 127.5, 1. / 127.5]
        self.num_threads = 4
        self.model = ncnn.Net()
        self.model = self.load_model(self.model, param_path, bin_path)
        self.align_face = align_face
        print('Finished loading face embedding model!')

    def load_model(self, model, model_param_path, model_bin_path):
        model.load_param(model_param_path)
        model.load_model(model_bin_path)
        return model

    def extract(self, image):
        ncnn_image = ncnn.Mat.from_pixels(image, ncnn.Mat.PixelType.PIXEL_BGR, image.shape[1], image.shape[0])
        ncnn_image.substract_mean_normalize(self.mean_vals, self.norm_vals)
        ex = self.model.create_extractor()
        ex.input("data", ncnn_image)
        ret, mat_out = ex.extract("683")
        ex.clear()
        feature = np.array(mat_out)
        embeddings = feature / np.linalg.norm(feature)
        return embeddings.tolist() # vector 

    def alignment(self, image, bboxes, landms):
        faces_aligns = self.align_face(image, landms[0])
        return faces_aligns
