import math,six,random,cv2
import numpy as np
from utils.bbox_utils.bbox_iou import bbox_iou
from utils.image import random_expand,resize,random_flip
from utils.bbox import translate_bbox,crop_bbox,resize_bbox,flip_bbox

def random_distort(img, brightness_delta=32, contrast_low=0.5, contrast_high=1.5, saturation_low=0.5, saturation_high=1.5, hue_delta=18):
    cv_img = img[::-1].transpose((1, 2, 0)).astype(np.uint8)

    def convert(img, alpha=1, beta=0):
        img = img.astype(float) * alpha + beta
        img[img < 0] = 0
        img[img > 255] = 255
        return img.astype(np.uint8)

    def brightness(cv_img, delta):
        if random.randrange(2):
            return convert(
                cv_img,
                beta=random.uniform(-delta, delta))
        else:
            return cv_img

    def contrast(cv_img, low, high):
        if random.randrange(2):
            return convert(
                cv_img,
                alpha=random.uniform(low, high))
        else:
            return cv_img

    def saturation(cv_img, low, high):
        if random.randrange(2):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            cv_img[:, :, 1] = convert(
                cv_img[:, :, 1],
                alpha=random.uniform(low, high))
            return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
        else:
            return cv_img

    def hue(cv_img, delta):
        if random.randrange(2):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            cv_img[:, :, 0] = (
                cv_img[:, :, 0].astype(int) +
                random.randint(-delta, delta)) % 180
            return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
        else:
            return cv_img

    cv_img = brightness(cv_img, brightness_delta)

    if random.randrange(2):
        cv_img = contrast(cv_img, contrast_low, contrast_high)
        cv_img = saturation(cv_img, saturation_low, saturation_high)
        cv_img = hue(cv_img, hue_delta)
    else:
        cv_img = saturation(cv_img, saturation_low, saturation_high)
        cv_img = hue(cv_img, hue_delta)
        cv_img = contrast(cv_img, contrast_low, contrast_high)

    return cv_img.astype(np.float32).transpose((2, 0, 1))[::-1]

def random_crop_with_bbox_constraints(img, bbox, min_scale=0.3, max_scale=1, max_aspect_ratio=2, constraints=None, max_trial=50, return_param=False):
    if constraints is None:
        constraints = (
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, 1),
        )

    _, H, W = img.shape
    params = [{
        'constraint': None, 'y_slice': slice(0, H), 'x_slice': slice(0, W)}]

    if len(bbox) == 0:
        constraints = []

    for min_iou, max_iou in constraints:
        if min_iou is None:
            min_iou = 0
        if max_iou is None:
            max_iou = 1

        for _ in six.moves.range(max_trial):
            scale = random.uniform(min_scale, max_scale)
            aspect_ratio = random.uniform(max(1 / max_aspect_ratio, scale * scale), min(max_aspect_ratio, 1 / (scale * scale)))
            crop_h = int(H * scale / np.sqrt(aspect_ratio))
            crop_w = int(W * scale * np.sqrt(aspect_ratio))

            crop_t = random.randrange(H - crop_h)
            crop_l = random.randrange(W - crop_w)
            crop_bb = np.array((
                crop_t, crop_l, crop_t + crop_h, crop_l + crop_w))

            iou = bbox_iou(bbox, crop_bb[np.newaxis])
            if min_iou <= iou.min() and iou.max() <= max_iou:
                params.append({
                    'constraint': (min_iou, max_iou),
                    'y_slice': slice(crop_t, crop_t + crop_h),
                    'x_slice': slice(crop_l, crop_l + crop_w)})
                break

    param = random.choice(params)
    img = img[:, param['y_slice'], param['x_slice']]

    if return_param:
        return img, param
    else:
        return img

def resize_with_random_interpolation(img, size, return_param=False):
    cv_img = img.transpose((1, 2, 0))

    inters = (
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    )
    inter = random.choice(inters)
    H, W = size
    cv_img = cv2.resize(cv_img, (W, H), interpolation=inter)

    # If input is a grayscale image, cv2 returns a two-dimentional array.
    if len(cv_img.shape) == 2:
        cv_img = cv_img[:, :, np.newaxis]

    img = cv_img.astype(np.float32).transpose((2, 0, 1))

    if return_param:
        return img, {'interpolation': inter}
    else:
        return img

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

class DataAugmentationTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, in_data):
        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = random_expand(img, fill=0, return_param=True)
            bbox = translate_bbox(bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        bbox, param = crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = random_flip(img, x_random=True, return_param=True)
        bbox = flip_bbox(bbox, (self.size, self.size), x_flip=params['x_flip'])

        return img, bbox, label

class CenterDetectionTransform:
    def __init__(self, insize, num_classes, downratio, dtype=np.float32) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.downratio = downratio
        self.size = insize
        self.dtype = dtype

    def __call__(self, in_data):
        img, bboxes, labels = in_data[:3]

        output_h = self.size // self.downratio
        output_w = self.size // self.downratio

        _, H, W, = img.shape
        img = resize(img, (self.size, self.size))
        bboxes = resize_bbox(bboxes, (H, W), (output_h, output_w))

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=self.dtype)
        dense_wh = np.zeros((2, output_h, output_w), dtype=self.dtype)
        dense_offset = np.zeros((2, output_h, output_w), dtype=self.dtype)
        dense_mask = np.zeros((2, output_h, output_w), dtype=self.dtype)

        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            w = bbox[3] - bbox[1]
            h = bbox[2] - bbox[0]

            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))

            center = np.array([
                (bbox[3] + bbox[1]) / 2, (bbox[2] + bbox[0]) / 2
            ], dtype=self.dtype)
            center_int = center.astype(np.int32)
            draw_umich_gaussian(hm[label], center_int, radius)
            dense_wh[0, center_int[1], center_int[0]] = w
            dense_wh[1, center_int[1], center_int[0]] = h
            dense_offset[0, center_int[1], center_int[0]] = (center - center_int)[0]
            dense_offset[1, center_int[1], center_int[0]] = (center - center_int)[1]
            dense_mask[0, center_int[1], center_int[0]] = 1.0
            dense_mask[1, center_int[1], center_int[0]] = 1.0

        return {
            'image': img,
            'hm': hm,
            'dense_wh': dense_wh, 'dense_mask': dense_mask, 'dense_offset': dense_offset,
        }