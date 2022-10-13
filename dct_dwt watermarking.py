import sys

import numpy as np
import pywt
import os
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct
from pprint import pprint


def convert_image(image_name: str, size: int) -> np.ndarray:
    img = Image.open('./pictures/' + image_name).resize((size, size), 1)
    img = img.convert('L')  # 8-bit pixels, black and white
    img.save('./dataset/' + image_name)

    image_array = np.array(img.getdata(), dtype=np.uintc).reshape((size, size))

    return image_array


class DigitalWatermarkDctDwt:
    def __init__(self, image_name: str, watermark_name: str, model: str = "haar", level: int = 1):
        self.image_array: np.ndarray = convert_image(image_name, 2048)
        self.watermark_array: np.ndarray = convert_image(watermark_name, 128)
        self.model: str = model
        self.level: int = level
        self.dct_array = None
        self.coeffs_image = None
        self.image_array_H = None

    def __process_coefficients(self, im_array: np.ndarray) -> list:
        """
        Decomposition into coefficients (cA, (cH, cV, cD))
        """
        coeffs = pywt.wavedec2(data=im_array, wavelet=self.model, level=self.level)
        coeffs_H = list(coeffs)

        return coeffs_H

    @staticmethod
    def __apply_dct(image_array: list) -> np.ndarray:
        size = len(image_array[0])
        all_subdct = np.empty((size, size))
        for i in range(0, size, 8):
            for j in range(0, size, 8):
                subpixels = image_array[i:i + 8, j:j + 8]
                subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
                all_subdct[i:i + 8, j:j + 8] = subdct
        return all_subdct

    def __embed_watermark(self):
        watermark_flat = self.watermark_array.ravel()  # сглаживание
        ind = 0
        print(len(self.dct_array))
        print(len(watermark_flat))
        for x in range(0, len(self.dct_array), 8):
            for y in range(0, len(self.dct_array), 8):
                if ind < len(watermark_flat):
                    subdct = self.dct_array[x:x + 8, y:y + 8]
                    subdct[5][5] = watermark_flat[ind]
                    self.dct_array[x:x + 8, y:y + 8] = subdct
                    ind += 1
        return self.dct_array

    def __inverse_dct(self) -> np.ndarray:
        size = len(self.dct_array[0])
        all_subidct = np.empty((size, size))
        for i in range(0, size, 8):
            for j in range(0, size, 8):
                subidct = \
                    idct(idct(self.dct_array[i:i + 8, j:j + 8].T, norm="ortho").T, norm="ortho")
                all_subidct[i:i + 8, j:j + 8] = subidct

        return all_subidct

    def __print_image_from_array(self, name: str) -> None:
        image_array_copy = self.image_array_H.clip(0, 255)
        image_array_copy = image_array_copy.astype("uint8")
        img = Image.fromarray(image_array_copy)
        img.save('./result/' + name)

    @staticmethod
    def __get_watermark(dct_watermarked_coeff: np.ndarray, watermark_size: int) -> np.ndarray:
        subwatermarks = []

        for x in range(0, len(dct_watermarked_coeff), 8):
            for y in range(0, len(dct_watermarked_coeff), 8):
                coeff_slice = dct_watermarked_coeff[x:x + 8, y:y + 8]
                subwatermarks.append(coeff_slice[5][5])

        watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)

        return watermark

    def __recover_watermark(self) -> None:
        coeffs_watermarked_image = self.__process_coefficients(self.image_array_H)
        dct_watermarked_coeff = self.__apply_dct(coeffs_watermarked_image[0])

        watermark_array = self.__get_watermark(dct_watermarked_coeff, 128)

        watermark_array = np.uint8(watermark_array)

        # Save result
        img = Image.fromarray(watermark_array)
        img.save('./result/recovered_watermark.jpg')

    def implementation_dw(self):
        self.coeffs_image = self.__process_coefficients(self.image_array)
        self.dct_array: np.ndarray = self.__apply_dct(self.coeffs_image[0])
        self.dct_array = self.__embed_watermark()
        self.coeffs_image[0]: np.ndarray = self.__inverse_dct()

    def recover_watermark(self):
        self.image_array_H: np.ndarray = pywt.waverec2(self.coeffs_image, self.model)
        self.__print_image_from_array('image_with_watermark_szhatie.jpg')

        self.__recover_watermark()





image = 'kot-vodolaz.jpg'
watermark = 'qr-code.png'
wm = DigitalWatermarkDctDwt(image, watermark)
wm.implementation_dw()
wm.recover_watermark()
