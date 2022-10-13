import sys

import numpy as np
import pywt
import os
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct
from pprint import pprint

current_path = str(os.path.dirname(__file__))

image = 'kot-vodolaz.jpg'
watermark = 'qr-code.png'

flag = True


def print_once(name: str) -> None:
    global flag
    if flag:
        print(name)
        flag = False


def save_image(array: np.ndarray, name: str) -> None:
    img = Image.fromarray(np.uint8(array))
    img.save(f'./learning/{name}.jpg')


def print_to_file(*args, **kwargs):
    for key in kwargs.keys():
        with open(f"./learning/{key}", "w") as f:
            f.write(str(kwargs[key]))


def convert_image(image_name: str, size: int) -> np.ndarray:
    img = Image.open('./pictures/' + image_name).resize((size, size), 1)
    img = img.convert('L')  # 8-bit pixels, black and white
    img.save('./dataset/' + image_name)

    image_array = np.array(img.getdata(), dtype=np.uintc).reshape((size, size))

    return image_array


def process_coefficients(im_array: np.ndarray, model: str, level: int) -> list:
    """
    Decomposition into coefficients (cA, (cH, cV, cD))
    """
    coeffs = pywt.wavedec2(data=im_array, wavelet=model, level=level)
    coeffs_H = list(coeffs)
    save_image(coeffs[0], "after_dvp")
    return coeffs_H


# def embed_mod2(coeff_image, coeff_watermark, offset=0):
#     for i in range(coeff_watermark.__len__()):
#         for j in range(coeff_watermark[i].__len__()):
#             coeff_image[i * 2 + offset][j * 2 + offset] = coeff_watermark[i][j]
#
#     return coeff_image
#
#
# def embed_mod4(coeff_image, coeff_watermark):
#     for i in range(coeff_watermark.__len__()):
#         for j in range(coeff_watermark[i].__len__()):
#             coeff_image[i * 4][j * 4] = coeff_watermark[i][j]
#
#     return coeff_image


def embed_watermark(watermark_array: np.ndarray,
                    orig_image: np.ndarray) -> np.ndarray:
    """
    watermark_array=array([[  0,   0,   0, ...,   0,   0,   0],
       [  0,   0,   0, ...,   0,   0,   0],
       [  0,   0,   0, ...,   0,   0,   0],
       ...,
       [  0,   0,   0, ..., 255, 255, 255],
       [  0,   0,   0, ..., 255, 255, 255],
       [  0,   0,   0, ..., 245, 245, 245]], dtype=uint32)

    orig_image:
[[ 3.08331250e+03  5.34939509e+00  5.56209557e+00 ...  1.52720600e-01  -2.61312593e+00  1.12935076e+00]
 [ 6.72960079e+01  1.39645451e+00 -4.19961162e-01 ...  0.00000000e+00   0.00000000e+00  0.00000000e+00]
 [-1.42925238e+01  8.52370663e+00  7.00825215e-02 ...  0.00000000e+00   0.00000000e+00  0.00000000e+00]
 ...
 [ 2.15285403e-01  6.86138328e-01  7.05735977e-01 ...  1.05761062e+00  -1.00193041e+00  1.11968634e+00]
 [-5.69217446e-01 -1.85851173e-01 -6.91941738e-01 ...  1.59094823e-01   4.26776695e-01 -1.87665139e-01]
 [-2.19759471e-01  6.30543481e-02  1.88837192e-01 ...  8.19958253e-01   3.50387893e-01  5.39628167e-01]]
    """

    '''
    Итерируемся по первым битам каждого байта пока не щакончатся биты секретного сообщения
    После первой итерации subdct =
 [[ 3.08331250e+03  5.34939509e+00  5.56209557e+00  5.41094513e-01 -4.37500000e-01 -6.02740838e-01  1.99136543e-01  6.07056323e-01]
 [ 6.72960079e+01  1.39645451e+00 -4.19961162e-01 -9.25750123e-02 -1.89956983e-01 -1.47425646e+00 -1.13287212e+00 -6.36375425e-01]
 [-1.42925238e+01  8.52370663e+00  7.00825215e-02  7.97634034e-01  3.40651414e-01 -1.03969587e-01  2.79029131e-01  9.60415796e-01]
 [ 6.26768677e+00 -3.46592471e-01 -8.58427394e-01 -1.89483462e+00 -5.49043613e-01  5.50206984e-01 -3.37034514e-01  9.70839023e-01]
 [-9.93750000e+00  8.85846966e-01 -1.49309698e-01  6.59531400e-01  3.12500000e-01 -9.83887922e-02  1.29495614e-01 -6.41278512e-01]
 [ 9.88674386e-01 -2.88972844e-02  4.32366046e-01  7.16535935e-02  2.78518134e-01 -1.22932331e-01 -1.33967095e-01 -5.10973421e-02]
 [-3.71247447e-01 -2.21323794e-01 -7.20970869e-01 -2.59074333e-01  3.32444152e-01 -3.02129703e-01 -1.95082521e-01  3.17586288e-01]
 [ 3.96491872e-02 -8.64928815e-01 -6.03297751e-01  1.78761330e-01  6.89579658e-01 -3.45167506e-02 -1.64735482e-01  1.21312446e-01]]
   
   а orig_image =
 [[ 3.08331250e+03  5.34939509e+00  5.56209557e+00 ...  1.52720600e-01
  -2.61312593e+00  1.12935076e+00]
 [ 6.72960079e+01  1.39645451e+00 -4.19961162e-01 ...  0.00000000e+00
   0.00000000e+00  0.00000000e+00]
 [-1.42925238e+01  8.52370663e+00  7.00825215e-02 ...  0.00000000e+00
   0.00000000e+00  0.00000000e+00]
 ...
 [ 2.15285403e-01  6.86138328e-01  7.05735977e-01 ...  1.05761062e+00
  -1.00193041e+00  1.11968634e+00]
 [-5.69217446e-01 -1.85851173e-01 -6.91941738e-01 ...  1.59094823e-01
   4.26776695e-01 -1.87665139e-01]
 [-2.19759471e-01  6.30543481e-02  1.88837192e-01 ...  8.19958253e-01
   3.50387893e-01  5.39628167e-01]]
    '''
    watermark_flat = watermark_array.ravel()  # сглаживание
    ind = 0
    subdct = None
    save_image(orig_image, "orig_image_start<embed_watermark>")
    for x in range(0, orig_image.__len__(), 8):
        for y in range(0, orig_image.__len__(), 8):
            if ind < watermark_flat.__len__():
                subdct = orig_image[x:x + 8, y:y + 8]
                subdct[5][5] = watermark_flat[ind]
                orig_image[x:x + 8, y:y + 8] = subdct
                ind += 1
    np.set_printoptions(edgeitems=20)
    print_to_file(subdct=subdct)
    print_to_file(orig_image=orig_image)
    print_to_file(watermark_flat=watermark_flat)
    save_image(subdct, "subd_finish<embed_watermark>")
    save_image(orig_image, "orig_image_finish<embed_watermark>")
    save_image(watermark_flat, "watermark_flat_finish<embed_watermark>")

    return orig_image


def apply_dct(image_array: list) -> np.ndarray:
    size = image_array[0].__len__()
    all_subdct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subpixels = image_array[i:i + 8, j:j + 8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i + 8, j:j + 8] = subdct
    return all_subdct


def inverse_dct(all_subdct: np.ndarray) -> np.ndarray:
    size = all_subdct[0].__len__()
    all_subidct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subidct = idct(idct(all_subdct[i:i + 8, j:j + 8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i + 8, j:j + 8] = subidct
    save_image(all_subdct, "inverse_dct")
    return all_subidct


def get_watermark(dct_watermarked_coeff: np.ndarray, watermark_size: int) -> np.ndarray:
    subwatermarks = []

    for x in range(0, dct_watermarked_coeff.__len__(), 8):
        for y in range(0, dct_watermarked_coeff.__len__(), 8):
            coeff_slice = dct_watermarked_coeff[x:x + 8, y:y + 8]
            subwatermarks.append(coeff_slice[5][5])

    watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)

    return watermark


def recover_watermark(image_array: np.ndarray, model='haar', level=1) -> None:
    coeffs_watermarked_image = process_coefficients(image_array, model, level=level)
    dct_watermarked_coeff = apply_dct(coeffs_watermarked_image[0])

    watermark_array = get_watermark(dct_watermarked_coeff, 128)  # todo: change

    watermark_array = np.uint8(watermark_array)

    # Save result
    img = Image.fromarray(watermark_array)
    img.save('./result/recovered_watermark.jpg')


def print_image_from_array(image_array: np.ndarray, name: str) -> None:
    image_array_copy = image_array.clip(0, 255)
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)
    img.save('./result/' + name)


def w2d():
    model = 'haar'
    level = 1
    image_array: np.ndarray = convert_image(image, 2048)
    watermark_array: np.ndarray = convert_image(watermark, 128)
    # image_array: np.ndarray = convert_image(image, 64)
    # watermark_array: np.ndarray = convert_image(watermark, 16)

    coeffs_image: list = process_coefficients(image_array, model, level=level)
    dct_array: np.ndarray = apply_dct(coeffs_image[0])
    dct_array: np.ndarray = embed_watermark(watermark_array, dct_array)
    coeffs_image[0]: np.ndarray = inverse_dct(dct_array)

    # reconstruction
    image_array_H: np.ndarray = pywt.waverec2(coeffs_image, model)
    print_image_from_array(image_array_H, 'image_with_watermark.jpg')

    # recover images
    recover_watermark(image_array=image_array_H, model=model, level=level)


if __name__ == '__main__':
    w2d()
