import numpy as np
import PIL.Image,cv2, random

def resize(img, size, interpolation=PIL.Image.BILINEAR):
    img = img.transpose((1, 2, 0))
    if interpolation == PIL.Image.NEAREST:
        cv_interpolation = cv2.INTER_NEAREST
    elif interpolation == PIL.Image.BILINEAR:
        cv_interpolation = cv2.INTER_LINEAR
    elif interpolation == PIL.Image.BICUBIC:
        cv_interpolation = cv2.INTER_CUBIC
    elif interpolation == PIL.Image.LANCZOS:
        cv_interpolation = cv2.INTER_LANCZOS4
    H, W = size
    img = cv2.resize(img, dsize=(W, H), interpolation=cv_interpolation)

    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img.transpose((2, 0, 1))

def random_expand(img, max_ratio=4, fill=0, return_param=False):
    if max_ratio <= 1:
        if return_param:
            return img, {'ratio': 1, 'y_offset': 0, 'x_offset': 0}
        else:
            return img

    C, H, W = img.shape

    ratio = random.uniform(1, max_ratio)
    out_H, out_W = int(H * ratio), int(W * ratio)

    y_offset = random.randint(0, out_H - H)
    x_offset = random.randint(0, out_W - W)

    out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = np.array(fill).reshape((-1, 1, 1))
    out_img[:, y_offset:y_offset + H, x_offset:x_offset + W] = img

    if return_param:
        param = {'ratio': ratio, 'y_offset': y_offset, 'x_offset': x_offset}
        return out_img, param
    else:
        return out_img

def random_flip(img, y_random=False, x_random=False, return_param=False, copy=False):
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img
