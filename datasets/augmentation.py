import itertools
import random
import numpy as np
import torch
import torchvision.transforms.functional as F


class CustomAugmentation(object):

    def __init__(self, vflip : bool, hflip : bool, affine : bool,
                adjust_brightness : bool, adjust_contrast : bool,
                adjust_saturation : bool, adjust_hue : bool, crop: bool):

        self.vflip = vflip
        self.hflip = hflip
        self.affine = affine
        self.adjust_brightness = adjust_brightness
        self.adjust_contrast = adjust_contrast
        self.adjust_saturation = adjust_saturation
        self.adjust_hue = adjust_hue
        self.crop = crop

    @staticmethod
    def from_id(id_augm):
        hyperspace = [[True, False] for _ in range(8)]
        hyperspace = list(itertools.product(*hyperspace))
        print("Total combinations: {}".format(len(hyperspace)))
        h = hyperspace[id_augm]
        return CustomAugmentation(*h)

    @staticmethod
    def get_affine_params(w, h):
        affine_angle_rot = random.randint(-5, 5)
        max_dx = 0.05 * w
        max_dy = 0.05 * h
        affine_translate = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        affine_scale = random.uniform(1.0, 1.05)
        affine_shear = random.uniform(-1.0, 1.0)

        return affine_angle_rot, affine_translate, \
               affine_scale, affine_shear

    @staticmethod
    def get_color_jitter_params(w, h):
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)
        hue_factor = random.uniform(-0.02, 0.02)
        return brightness_factor, contrast_factor, \
               saturation_factor, hue_factor

    @staticmethod
    def get_crop_params(w, h):
        crop_size = random.choice([0.75, 0.8, 0.85, 0.9])
        th, tw = (int(crop_size * h), int(crop_size * w))
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image):

        w, h = image.shape[1], image.shape[2]

        affine_angle_rot, affine_translate, affine_scale, \
                affine_shear = self.get_affine_params(w, h)

        brightness_factor, contrast_factor, \
            saturation_factor, hue_factor = self.get_color_jitter_params(w, h)

        i, j, th, tw = self.get_crop_params(w, h)

        def apply_bool(val):
            return val and random.choice([True, False])

        apply_crop = apply_bool(self.crop)
        apply_vflip = apply_bool(self.vflip)
        apply_hflip = apply_bool(self.hflip)
        apply_adjust_brightness = apply_bool(self.adjust_brightness)
        apply_adjust_contrast = apply_bool(self.adjust_contrast)
        apply_adjust_saturation = apply_bool(self.adjust_saturation)
        apply_adjust_hue = apply_bool(self.adjust_hue)
        apply_affine = apply_bool(self.affine)

        # image = deepcopy(image_original)
        image = F.to_pil_image(torch.from_numpy(image))


        if apply_crop:
            image = F.crop(image, i, j, th, tw)
            image = F.resize(image, (w, h))
        if apply_vflip:
            image = F.vflip(image)
        if apply_hflip:
            image = F.hflip(image)
        if apply_adjust_brightness:
            image = F.adjust_brightness(image, brightness_factor)
        if apply_adjust_contrast:
            image = F.adjust_contrast(image, contrast_factor)
        if apply_adjust_saturation:
            image = F.adjust_saturation(image, saturation_factor)
        if apply_adjust_hue:
            image = F.adjust_hue(image, hue_factor)
        if apply_affine:
            image = F.affine(image, affine_angle_rot,
                             affine_translate,
                             affine_scale, affine_shear)

        image = F.to_tensor(image).numpy() #.transpose((1, 2, 0))

        return image