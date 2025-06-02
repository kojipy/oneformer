import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

MODEL = "shi-labs/oneformer_cityscapes_swin_large"


def parse_args():
    parser = argparse.ArgumentParser(description="OneFormer Segmentation")
    parser.add_argument("image", type=Path, help="image path to predict")
    parser.add_argument(
        "--model",
        type=str,
        default="shi-labs/oneformer_cityscapes_swin_large",
        help="Model name or path",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["panoptic", "semantic", "instance"],
        default="panoptic",
        help="Segmentation task",
    )
    return parser.parse_args()


class OneFormer:
    def __init__(self, model_name: str, task: str = "panoptic"):
        self._processor = OneFormerProcessor.from_pretrained(model_name)
        self._model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self._task = task

    def predict(self, image: Image.Image):
        inputs = self._preprocess(image)

        with torch.no_grad():
            outputs = self._model(**inputs)

        outputs = self._postprocess(outputs, target_sizes=[(image.height, image.width)])

        return outputs

    def _preprocess(self, image: Image.Image):
        inputs = self._processor(image, [self._task], return_tensors="pt")
        return inputs

    def _postprocess(self, outputs, target_sizes):
        if self._task == "panoptic":
            return self._processor.post_process_panoptic_segmentation(
                outputs, target_sizes=target_sizes
            )
        elif self._task == "semantic":
            return self._processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
        elif self._task == "instance":
            return self._processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes
            )
        else:
            raise ValueError(f"Unknown task: {self._task}")


def _create_color_map(segments_info):
    color_map = {}
    unique_label_ids = set([info["label_id"] for info in segments_info])
    for label_id in unique_label_ids:
        color_map[label_id] = random.sample(range(50, 230), k=3)
    return color_map


def plot(image, predict_mask):
    color_map = _create_color_map(predict_mask["segments_info"])

    img_arr = np.array(image.convert("RGB"))
    panoptic_segmentation_mask = predict_mask["segmentation"]

    canvas = np.zeros_like(img_arr)
    mask_numpy = panoptic_segmentation_mask.numpy()
    for info in predict_mask["segments_info"]:
        canvas[np.where(mask_numpy == info["id"])] = color_map[info["label_id"]]

        # インスタンスごとの輪郭を計算して赤枠で囲む
        mask = np.zeros_like(mask_numpy, dtype=np.uint8)
        mask[np.where(mask_numpy == info["id"])] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 0, 255), thickness=2)

    return canvas


def main():
    args = parse_args()
    oneformer = OneFormer(args.model)

    image = Image.open(args.image)
    outputs = oneformer.predict(image)
    predicted_panoptic = outputs[0]
    result = plot(image, predicted_panoptic)

    cv2.imwrite("mask.png", result)


if __name__ == "__main__":
    main()
