import cv2
import numpy as np
import os
from scipydct2 import SciPyDCT2
from utils import apply_frequency_cutoff

class ImageCompressor:
    def __init__(self, file_path, F, d):
        self.file_path = file_path
        self.F = F
        self.d = d
        self.original_filename = os.path.splitext(os.path.basename(file_path))[0]
        self.compressed_image_filename = f"{self.original_filename}_f{F}_d{d}.jpg"
        self.image = None
        self.compressed_image = None

    def load_image(self):
        self.image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image from {self.file_path}")

    def compress(self):
        if self.image is None:
            self.load_image()

        height, width = self.image.shape
        self.compressed_image = np.zeros_like(self.image)
        dct_engine = SciPyDCT2()

        for i in range(0, height, self.F):
            for j in range(0, width, self.F):
                block = self.image[i:i+self.F, j:j+self.F]
                if block.shape[0] == self.F and block.shape[1] == self.F:
                    dct_block = dct_engine.dct2(block)
                    dct_block = apply_frequency_cutoff(dct_block, self.d)
                    idct_block = dct_engine.idct2(dct_block)
                    self.compressed_image[i:i+self.F, j:j+self.F] = np.clip(idct_block, 0, 255)

        return self.compressed_image

    def save_compressed_image(self, save_dir):
        if self.compressed_image is None:
            raise ValueError("Compressed image not available. Please run compress() first.")

        save_path = os.path.join(save_dir, self.compressed_image_filename)
        cv2.imwrite(save_path, self.compressed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return save_path
