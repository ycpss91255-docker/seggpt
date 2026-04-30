"""SegGPT service.

Direct port of ``generative_services.services.seggpt_service`` with
import paths rewritten for the new ``seggpt.runtime`` package. The
target / prompt / reset state machine, the instance / semantic mode
helpers, and ``SegGPTDaemonService`` (stateless variant) are all
preserved verbatim so we stay in step with upstream's behaviour.
"""
from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from yacs.config import CfgNode as CN

from seggpt.runtime.services.abstract_service import AbstractService
from seggpt.runtime.services.import_modules import import_torch as torch
from seggpt.runtime.services.import_modules import (
    import_torchvision_transforms_functional as F,
)
from seggpt.runtime.services.import_self import import_seggpt_model as seggpt_model
from seggpt.runtime.services.utils import torch_use_cuda
from seggpt.runtime.utils.logger import logi_print
from seggpt.runtime.utils.naming import to_camel_case, to_snake_case
from seggpt.runtime.utils.tools import load_yaml
from seggpt.runtime.utils.types import PathLike, class_property

_keywords = [
    'SegGPT',
    'seggpt',
    to_snake_case('SegGPT'),
    to_camel_case('SegGPT'),
    to_snake_case('SegGPTService'),
    to_camel_case('SegGPTService'),
]
_daemon_keywords = [
    'SegGPTDaemon',
    'seggpt_daemon',
    to_snake_case('SegGPTDaemon'),
    to_camel_case('SegGPTDaemon'),  
]
_seg_types = ['instance', 'semantic', 'panoptic']
_imagenet_mean = np.array([0.485, 0.456, 0.406])
_imagenet_std = np.array([0.229, 0.224, 0.225])


def _preprocess(image: np.ndarray, size: List[int, int]) -> torch.Tensor:
    """Resize the image."""
    return F.normalize(
        F.to_tensor(
            F.resize(
                F.to_pil_image(image),
                size
            )
        ),
        mean=_imagenet_mean,
        std=_imagenet_std
    )


def _check_image_and_mask(
        images: Union[np.ndarray, List[np.ndarray]],
        masks: Union[np.ndarray, List[np.ndarray]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if isinstance(images, np.ndarray) and images.ndim == 3:
        images = [images]
    if len(images) != len(masks):
        if len(images) != 1:
            raise ValueError(f"Invalid number of images: {len(images)} != {len(masks)}")
        masks = [masks]
    images = list(images)
    masks = list(masks)
    for ind, image in enumerate(images):
        mask = masks[ind]
        if image.ndim != 3:
            raise ValueError(f"The shape of image {ind} is invalid: {image.shape}, which should be (H, W, 3).")
        if mask.ndim == 2:
            masks[ind] = mask = np.array([mask])
        elif mask.ndim != 3:
            raise ValueError(f"The shape of mask {ind} is invalid: {mask.shape}, which should be (C, H, W).")
        if image.shape[:2] != mask.shape[1:]:
            raise ValueError(f"The resolution of the {ind} pair is invalid: {image.shape[:2]} != {mask.shape[1:]}")
        if mask.dtype == np.float32:
            if mask.min() < 0 or mask.max() > 1:
                raise ValueError(f"The value range of mask {ind} is invalid: [{mask.min()}, {mask.max()}]")
        elif mask.dtype == bool:
            masks[ind] = mask.astype(np.float32)
        elif mask.dtype != np.uint8:
            raise ValueError(f"The data type of mask {ind} is invalid: {mask.dtype}")
    return images, masks


def _define_colors_per_location_mean_sep(num) -> np.ndarray:
    num_sep_per_channel = int((num - 1) ** (1 / 3)) + 1
    separation_per_channel = 256 // num_sep_per_channel
    rg_separation = num_sep_per_channel ** 2
    total_size = num_sep_per_channel ** 3
    color_list: List[Optional] = [None] * total_size
    for location in range(total_size):
        num_seq_r = location // rg_separation
        num_seq_g = (location % rg_separation) // num_sep_per_channel
        num_seq_b = location % num_sep_per_channel
        assert (num_seq_r <= num_sep_per_channel and
                num_seq_g <= num_sep_per_channel and
                num_seq_b <= num_sep_per_channel)
        channel_r = 255 - num_seq_r * separation_per_channel
        channel_g = 255 - num_seq_g * separation_per_channel
        channel_b = 255 - num_seq_b * separation_per_channel
        assert 0 <= channel_r <= 255 and 0 <= channel_g <= 255 and 0 <= channel_b <= 255
        # assert (channel_r, channel_g, channel_b) not in color_list
        color_list[location] = [channel_r, channel_g, channel_b]
    if [0, 0, 0] not in color_list:
        color_list.append([0, 0, 0])
    return np.array(color_list, dtype=np.uint8)


def _define_colors_per_location_norm_pos(global_step=4, local_step=20) -> np.ndarray:
    num_location_r = global_step ** 2
    sep_r = 256 // num_location_r
    sep_gb = 256 // local_step
    # sep_gb = 256 // num_location_gb + 1  # +1 for bigger sep in gb
    total_step = global_step * local_step
    color_list: List[Optional] = [None] * (total_step ** 2)
    for global_x, global_y in product(range(global_step), range(global_step)):
        num_seq_r = global_y * global_step + global_x
        channel_r = 255 - num_seq_r * sep_r
        for local_x, local_y in product(range(local_step), range(local_step)):
            channel_g = 255 - local_y * sep_gb
            channel_b = 255 - local_x * sep_gb
            assert 0 <= channel_r <= 255 and 0 <= channel_g <= 255 and 0 <= channel_b <= 255
            absolute_x = global_x * local_step + local_x
            absolute_y = global_y * local_step + local_y
            ind = absolute_y * total_step + absolute_x
            assert color_list[ind] is None
            color_list[ind] = [channel_r, channel_g, channel_b]
    if [0, 0, 0] not in color_list:
        color_list.append([0, 0, 0])
    return np.array(color_list, dtype=np.uint8)


def _sort_masks_by_area(masks: np.ndarray) -> List[int]:
    """Sort the masks by area."""
    num = masks.shape[0]
    areas = masks.sum(axis=(1, 2))
    sorted_index_list = sorted(range(num), key=lambda x: areas[x])
    return sorted_index_list


def _convert_mask_to_color_by_class(masks: np.ndarray, color_list: np.ndarray, class_id: np.ndarray) -> np.ndarray:
    """Convert the mask to color by class."""
    if masks.dtype != bool:
        masks = masks > 0
    sorted_index_list = _sort_masks_by_area(masks)
    color = np.zeros(list(masks.shape[1:]) + [3], dtype=np.uint8)
    for ind in sorted_index_list:
        if class_id[ind] >= 0:
            color[masks[ind]] = color_list[class_id[ind]]
    return color


def _convert_mask_by_class(masks: np.ndarray, class_id: np.ndarray, max_category: int) -> np.ndarray:
    if masks.dtype != bool:
        masks = masks > 0
    sorted_index_list = _sort_masks_by_area(masks)
    new_masks = np.zeros([max_category] + list(masks.shape[1:]), dtype=np.uint8)
    for ind in sorted_index_list:
        if class_id[ind] >= 0:
            new_masks[class_id[ind]][masks[ind]] = 255
    return new_masks


def center_of_mass(mask: np.ndarray, esp: float = 1e-6):
    """Calculate the centroid coordinates of the mask.

    Args:
        mask (np.ndarray): The mask to be calculated, shape (h, w).
        esp (float): Avoid dividing by zero. Default: 1e-6.

    Returns:
        The centroid normalized coordinates of the mask, shape (2,).
    """
    imh, imw = mask.shape
    grid_h = np.arange(imh)[:, None]
    grid_w = np.arange(imw)[None, :]
    normalizer = mask.sum().astype("float").clip(min=esp)
    center_h = (mask * grid_h).sum() / normalizer
    center_w = (mask * grid_w).sum() / normalizer
    return center_w / imw, center_h / imh


def _convert_mask_to_color_by_pos(masks: np.ndarray, color_list: np.ndarray, step: int) -> np.ndarray:
    if masks.dtype != bool:
        masks = masks > 0
    sorted_index_list = _sort_masks_by_area(masks)
    color = np.zeros(list(masks.shape[1:]) + [3], dtype=np.uint8)
    lstep = step - 1
    for ind in sorted_index_list:
        mask = masks[ind]
        center_x, center_y = center_of_mass(mask)
        ind = int(center_y * lstep) * step + int(center_x * lstep)
        color[mask] = color_list[ind]
    return color


def _check_class_id(
        masks: List[np.ndarray],
        class_id: Optional[List[np.ndarray]] = None
) -> Tuple[int, List[np.ndarray]]:
    if class_id is None:
        max_category = masks[0].shape[0]

        if any(m.shape[0] != max_category for m in masks):
            raise ValueError(f"The number of categories of masks is not equal: {[m.shape[0] for m in masks]}")

        class_id = [np.arange(max_category)] * len(masks)
    else:
        max_category = max(c.max() for c in class_id) + 1
    return max_category, class_id


class SegGPTService(AbstractService, keywords=_keywords):
    """SegGPT Service."""

    def __init__(
            self,
            config_path: PathLike,
            checkpoint_path: PathLike,
            # global_step_for_panoptic_task: int = 4,
            # local_step_for_panoptic_task: int = 20,
            **kwargs: Any
    ):
        """Initialize the service.

        Args:
            config_path (PathLike): The config path.
            checkpoint_path (PathLike): The checkpoint path.
            **kwargs (Any): The other arguments.
        """
        super().__init__(**kwargs)
        model_cfg = load_yaml(config_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = seggpt_model.SegGPT(**model_cfg)
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logi_print(f"Loaded checkpoint from {checkpoint_path}: {msg}")
        model.eval()
        self._device = torch_use_cuda()
        self._target_size = [model_cfg.img_size[0] // 2, model_cfg.img_size[1]]  # [H, W]
        self._predictor = seggpt_model.SegGPTPredictor(
            model=model,
            pixel_mean=_imagenet_mean,
            pixel_std=_imagenet_std,
            device=self._device,
        )
        self._processed_image = None
        self._processed_mask = torch.zeros([1] + self._target_size + [3], dtype=torch.uint8, device=self._device)
        self._masked_pos = torch.zeros(model.patch_embed.num_patches)
        self._masked_pos[(model.patch_embed.num_patches // 2):] = 1
        self._masked_pos = self._masked_pos.unsqueeze(dim=0).to(self._device)
        # self._panoptic_colors = _define_colors_per_location_norm_pos(
        #     global_step=global_step_for_panoptic_task,
        #     local_step=local_step_for_panoptic_task,
        # )
        # self._panoptic_step = global_step_for_panoptic_task * local_step_for_panoptic_task
        self._img_h = None
        self._img_w = None

    def reset(self) -> SegGPTService:
        # pylint: disable=arguments-differ,duplicate-code
        """Reset the service.

        Returns:
            The instance of GroundingDINOService.
        """
        self._processed_image = None
        self._img_h = None
        self._img_w = None
        return self


    def _preprocess_image(self, image: np.ndarray, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        """Preprocess the image."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = cv2.resize(image, (self._target_size[1], self._target_size[0]), interpolation=interpolation)
        return image

    def target(self, image: np.ndarray) -> SegGPTService:
        # pylint: disable=arguments-differ
        """Set the target.

        Args:
            image (np.ndarray): The target image in RGB format, which the shape is `(H, W, 3)`.

        Returns:
            The instance of GroundingDINOService.
        """
        if image.ndim != 3 and image.shape[-1] != 3:
            raise ValueError(f"The shape of image is invalid: {image.shape}, which should be (H, W, 3).")
        self._img_h, self._img_w = image.shape[:2]
        image = self._preprocess_image(image)[None]
        self._processed_image = torch.as_tensor(image, dtype=torch.uint8, device=self._device)
        return self

    def _instance_mode(
            self,
            images: np.ndarray,
            masks: List[np.ndarray],
            class_id: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        max_category, class_id = _check_class_id(masks, class_id)
        num = len(images)
        masks = np.array([_convert_mask_by_class(m, i, max_category) for i, m in zip(class_id, masks)])
        # seg_type = np.zeros(shape=[num, 1], dtype=np.uint8)
        seg_type = np.ones(shape=[num, 1], dtype=np.uint8)
        torch_images = torch.as_tensor(images, dtype=torch.uint8, device=self._device)
        torch_seg_type = torch.as_tensor(seg_type, dtype=torch.uint8, device=self._device)
        merge_between_batch = 0 if num > 1 else -1
        res_hw = [self._img_h, self._img_w]
        results = []

        for cid in range(max_category):
            mask = np.repeat(masks[:, cid, :, :, None], 3, axis=-1)
            pred = self._predictor.forward(
                self._processed_image,
                self._processed_mask,
                torch_images,
                torch.as_tensor(mask, dtype=torch.uint8, device=self._device),
                res_hw=res_hw,
                bool_masked_pos=self._masked_pos,
                seg_type=torch_seg_type,
                merge_between_batch=merge_between_batch,
            )
            results.append(pred.detach().cpu().numpy()[0].mean(-1) > 85)  # 255 / 3 = 85

        return np.array(results), np.arange(max_category)

    def _semantic_mode(
            self,
            images: np.ndarray,
            masks: List[np.ndarray],
            class_id: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        max_category, class_id = _check_class_id(masks, class_id)
        colors = _define_colors_per_location_mean_sep(max_category)
        num = len(images)
        masks = np.array([_convert_mask_to_color_by_class(m, colors, i) for i, m in zip(class_id, masks)])
        seg_type = np.zeros(shape=[num, 1], dtype=np.uint8)
        # seg_type = np.ones(shape=[num, 1], dtype=np.uint8)
        torch_images = torch.as_tensor(images, dtype=torch.uint8, device=self._device)
        torch_seg_type = torch.as_tensor(seg_type, dtype=torch.uint8, device=self._device)
        merge_between_batch = 0 if num > 1 else -1
        res_hw = [self._img_h, self._img_w]
        pred = self._predictor.forward(
            self._processed_image,
            self._processed_mask,
            torch_images,
            torch.as_tensor(masks, dtype=torch.uint8, device=self._device),
            res_hw=res_hw,
            bool_masked_pos=self._masked_pos,
            seg_type=torch_seg_type,
            merge_between_batch=merge_between_batch,
        )
        pred_np = pred.detach().cpu().numpy()[0]
        dist = np.abs(pred_np[:, :, None, :] - colors[None, None, :max_category, :]).sum(-1)
        seg = dist.argmin(-1)
        results = np.array([seg == i for i in range(max_category)])
        return np.array(results), np.arange(max_category)

    #
    # def _panoptic_mode(
    #         self,
    #         images: np.ndarray,
    #         masks: List[np.ndarray],
    #         class_id: Optional[List[np.ndarray]] = None
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     # evaluate instance mode first
    #     num = len(images)
    #     inst_masks = np.array([_convert_mask_to_color_by_pos(m, self._panoptic_colors, self._panoptic_step)
    #                            for m in masks])
    #     inst_seg_type = np.ones(shape=[num, 1], dtype=np.uint8)
    #     torch_images = torch.as_tensor(images, dtype=torch.uint8, device=self._device)
    #     merge_between_batch = 0 if num > 1 else -1
    #     res_hw = [self._img_h, self._img_w]
    #     pred = self._predictor.forward(
    #         self._processed_image,
    #         self._processed_mask,
    #         torch_images,
    #         torch.as_tensor(inst_masks, dtype=torch.uint8, device=self._device),
    #         res_hw=res_hw,
    #         bool_masked_pos=self._masked_pos,
    #         seg_type=torch.as_tensor(inst_seg_type, dtype=torch.uint8, device=self._device),
    #         merge_between_batch=merge_between_batch,
    #     )
    #     pred_np = pred.detach().cpu().numpy()[0]
    #     inst_dist = np.abs(pred_np[:, :, None, :] - self._panoptic_colors[None, None, :, :]).sum(-1)  # (H, W, N)
    #     inst_seg = inst_dist.argmin(-1)
    #     inst_masks = np.array([inst_seg == i for i in range(len(self._panoptic_colors))])  # (N, H, W)
    #
    #     if class_id is None:
    #         return inst_masks, np.arange(inst_masks.shape[0])
    #
    #     del inst_seg, inst_dist, pred_np, pred, inst_seg_type
    #     max_category = max(c.max() for c in class_id) + 1  # K
    #     colors = _define_colors_per_location_mean_sep(max_category)
    #     sema_masks = np.array([_convert_mask_to_color_by_class(m, colors, i) for i, m in zip(class_id, masks)])
    #     sema_seg_type = np.zeros(shape=[num, 1], dtype=np.uint8)
    #     pred = self._predictor.forward(
    #         self._processed_image,
    #         self._processed_mask,
    #         torch_images,
    #         torch.as_tensor(sema_masks, dtype=torch.uint8, device=self._device),
    #         res_hw=res_hw,
    #         bool_masked_pos=self._masked_pos,
    #         seg_type=torch.as_tensor(sema_seg_type, dtype=torch.uint8, device=self._device),
    #         merge_between_batch=merge_between_batch,
    #     )
    #     pred_np = pred.detach().cpu().numpy()[0]
    #     sema_dist = np.abs(pred_np[:, :, None, :] - colors[None, None, :, :]).sum(-1)  # (H, W, K)
    #     sema_score = 1. - sema_dist / sema_dist.max()
    #     class_probs = np.einsum('nhw,hwk->nk', inst_masks, sema_score)  # (N, K)
    #     pred_class_id = class_probs.argmax(-1)  # (N,)
    #     pred_class_id[pred_class_id >= max_category] = -1
    #     return inst_masks, pred_class_id

    def prompt(  # pylint: disable=arguments-differ
            self,
            images: List[np.ndarray],
            masks: List[np.ndarray],
            class_id: Optional[List[np.ndarray]] = None,
            segmentation_mode: str = 'instance',
    ) -> Dict[str, Any]:
        """Prompt the service.

        Args:
            images (List[np.ndarray]): The list of prompting images in RGB format,
                which the shape of each image is `(H, W, 3)`.
            masks (List[np.ndarray]): The list of prompting masks, which the shape of each mask is `(C, H, W)`.
                If the shape of each mask is `(H, W)`, the service will expand the dimension to `(1, H, W)`.
                The value of each pixel should be [0, 1] in float format, or be [0, 255] in uint8 format.
            class_id (Optional[List[np.ndarray]]): The list of class id, which the shape of each class id is `(C,)`.
                If this argument is None, the service will require the number of categories of each mask is equal.
                Otherwise, the service will require the number of categories of each mask is equal to the length of
                the corresponding class id.
                if the value in class id is -1 (or <0), the service will ignore the corresponding mask.
            segmentation_mode (str): Whether the task is instance, or semantic, or panoptic.

        Returns:
            The results of the current prompts, where the keys are:
                ```
                `mask` (np.ndarray): The predicted mask, which the shape is `(C, H, W)`.
                ```

        """
        if self._processed_image is None:
            raise RuntimeError("Please set the target first.")
        try:
            mode = _seg_types.index(segmentation_mode.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid segmentation mode: {segmentation_mode}, which should be one of {_seg_types}"
            ) from err

        images, masks = _check_image_and_mask(images, masks)
        images = np.array([self._preprocess_image(image) for image in images])
        masks = [np.array([self._preprocess_image(c, interpolation=cv2.INTER_NEAREST) for c in m]) for m in masks]
        # print(images.shape)
        # print([m.shape for m in masks])

        if mode == 0:
            new_mask, new_id = self._instance_mode(images, masks, class_id)
        elif mode == 1:
            # raise RuntimeError("Semantic mode is not supported yet.")
            new_mask, new_id = self._semantic_mode(images, masks, class_id)
        else:
            raise RuntimeError("Panoptic mode is not supported yet.")
            # new_mask, new_id = self._panoptic_mode(images, masks, class_id)
        return {'mask': new_mask, 'class_id': new_id}

    @property
    def output_keys(self) -> set[str]:
        """Get the name list of the outputs of the service.

        Returns:
            The name list of the outputs returned by the prompt function.
        """
        return {'mask', 'class_id'}

    @class_property
    @classmethod
    def default_config(cls) -> CN:
        """Get the default configuration of the service.

        Returns:
            The default configuration of the service.
        """
        cfg = super().default_config
        cfg.config_path = 'config/model/seggpt_vit_large.yaml'
        cfg.checkpoint_path = 'model/seggpt_vit_large.pth'
        return cfg


class SegGPTDaemonService(SegGPTService, keywords=_daemon_keywords):
    """SegGPT Daemon Service."""

    def __init__(self, **kwargs: Any):
        """Initialize the service.

        Args:
            **kwargs (Any): The other arguments.
        """
        super().__init__(**kwargs)

    def prompt(
        self,
        target_image: np.ndarray,
        prompt_images: List[np.ndarray],
        prompt_masks: List[np.ndarray],
        segmentation_mode='instance',
        class_id=None,
    ):
        """Predict"""
        if target_image.ndim != 3 and target_image.shape[-1] != 3:
            raise ValueError(f"The shape of image is invalid: {target_image.shape}, which should be (H, W, 3).")
        _img_h, _img_w = target_image.shape[:2]
        target_image = self._preprocess_image(target_image)[None]
        target_image = torch.as_tensor(target_image, dtype=torch.uint8, device=self._device)

        try:
            mode = _seg_types.index(segmentation_mode.lower())
        except ValueError as err:
            raise ValueError(
                f"Invalid segmentation mode: {segmentation_mode}, which should be one of {_seg_types}"
            ) from err

        prompt_images, prompt_masks = _check_image_and_mask(prompt_images, prompt_masks)
        prompt_images = np.array([self._preprocess_image(image) for image in prompt_images])
        prompt_masks = [np.array([self._preprocess_image(c, interpolation=cv2.INTER_NEAREST) for c in m]) for m in prompt_masks]

        if mode == 0:
            new_mask, new_id = self._daemon_instance_mode(
                target_image=target_image,
                prompt_images=prompt_images,
                prompt_masks=prompt_masks,
                img_h=_img_h,
                img_w=_img_w,
                class_id=class_id
            )
        elif mode == 1:
            new_mask, new_id = self._dameon_semantic_mode(
                target_image=target_image,
                prompt_images=prompt_images,
                prompt_masks=prompt_masks,
                img_h=_img_h,
                img_w=_img_w,
                class_id=class_id
            )
        else:
            raise RuntimeError("Panoptic mode is not supported yet.")
            # new_mask, new_id = self._panoptic_mode(images, masks, class_id)
        return {'mask': new_mask, 'class_id': new_id}

    def _dameon_semantic_mode(
            self,
            target_image,
            prompt_images,
            prompt_masks,
            img_h,
            img_w,
            class_id: Optional[List[np.ndarray]] = None
    ):
        max_category, class_id = _check_class_id(prompt_masks, class_id)
        colors = _define_colors_per_location_mean_sep(max_category)
        num = len(prompt_images)
        prompt_masks = np.array([_convert_mask_to_color_by_class(m, colors, i) for i, m in zip(class_id, prompt_masks)])
        seg_type = np.zeros(shape=[num, 1], dtype=np.uint8)
        # seg_type = np.ones(shape=[num, 1], dtype=np.uint8)
        torch_images = torch.as_tensor(prompt_images, dtype=torch.uint8, device=self._device)
        torch_seg_type = torch.as_tensor(seg_type, dtype=torch.uint8, device=self._device)
        merge_between_batch = 0 if num > 1 else -1
        res_hw = [img_h, img_w]
        pred = self._predictor.forward(
            target_image,
            self._processed_mask,
            torch_images,
            torch.as_tensor(prompt_masks, dtype=torch.uint8, device=self._device),
            res_hw=res_hw,
            bool_masked_pos=self._masked_pos,
            seg_type=torch_seg_type,
            merge_between_batch=merge_between_batch,
        )
        pred_np = pred.detach().cpu().numpy()[0]
        dist = np.abs(pred_np[:, :, None, :] - colors[None, None, :max_category, :]).sum(-1)
        seg = dist.argmin(-1)
        results = np.array([seg == i for i in range(max_category)])
        return np.array(results), np.arange(max_category)


    def _daemon_instance_mode(
            self,
            target_image,
            prompt_images,
            prompt_masks,
            img_h,
            img_w,
            class_id: Optional[List[np.ndarray]] = None
    ):
        max_category, class_id = _check_class_id(prompt_masks, class_id)
        num = len(prompt_images)
        prompt_masks = np.array([_convert_mask_by_class(m, i, max_category) for i, m in zip(class_id, prompt_masks)])
        # seg_type = np.zeros(shape=[num, 1], dtype=np.uint8)
        seg_type = np.ones(shape=[num, 1], dtype=np.uint8)
        torch_images = torch.as_tensor(prompt_images, dtype=torch.uint8, device=self._device)
        torch_seg_type = torch.as_tensor(seg_type, dtype=torch.uint8, device=self._device)
        merge_between_batch = 0 if num > 1 else -1

        res_hw = [img_h, img_w]
        results = []

        for cid in range(max_category):
            mask = np.repeat(prompt_masks[:, cid, :, :, None], 3, axis=-1)
            pred = self._predictor.forward(
                target_image,
                self._processed_mask,
                torch_images,
                torch.as_tensor(mask, dtype=torch.uint8, device=self._device),
                res_hw=res_hw,
                bool_masked_pos=self._masked_pos,
                seg_type=torch_seg_type,
                merge_between_batch=merge_between_batch,
            )
            results.append(pred.detach().cpu().numpy()[0].mean(-1) > 85)  # 255 / 3 = 85

        return np.array(results), np.arange(max_category)


    def target(self):
        return self