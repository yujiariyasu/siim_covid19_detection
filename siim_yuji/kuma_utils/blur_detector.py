import numpy as np
import cv2

class BlurDetector:
    def __init__(self,
                 criteria: float = 12.5) -> None:
        self.criteria = criteria

    def classify_image(self, image_file_path):
        image_array = cv2.imread(image_file_path)
        h_margin = 150
        w_margin = 250

        height = image_array.shape[0]
        width = image_array.shape[1]

        target_area = image_array[h_margin: height -
                                  h_margin, w_margin: width - w_margin]

        variance = cv2.Laplacian(target_area, cv2.CV_64F).var()
        return variance

        # if variance < self.criteria:
        #     flag = 1
        # else:
        #     flag = 0
        # return flag
