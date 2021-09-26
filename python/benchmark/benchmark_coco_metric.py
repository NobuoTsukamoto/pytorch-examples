#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TorchVision Benchmark.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import csv
import json
import os
import time
from functools import partial

import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.io import read_image
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    keypointrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes


def get_minival_ids(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    ret = []
    for line in lines:
        ret.append(int(line))
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory containing validation set TFRecord files.",
    )
    parser.add_argument(
        "--annotation_path", type=str, help="Path that contains COCO annotations"
    )
    parser.add_argument(
        "--allowlist_file",
        type=str,
        help="File with COCO image ids to preprocess, one on each line.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--display_every",
        type=int,
        default=1000,
        help="Number of iterations executed between two consecutive display of metrics",
    )
    parser.add_argument(
        "--num_warmup_iterations",
        type=int,
        default=50,
        help="Number of initial iterations skipped from timing",
    )
    parser.add_argument(
        "--target_duration",
        type=int,
        default=None,
        help="If set, script will run for specified number of seconds.",
    )
    parser.add_argument(
        "--model_name", help="Class name of torchvision.", required=True
    )
    parser.add_argument("--read_im", help="cv, pil, torch", default="cv")
    parser.add_argument("--output", default="output.csv")
    args = parser.parse_args()

    model = globals()[args.model_name](pretrained=True)
    # model = fasterrcnn_resnet50_fpn(pretrained=True)
    # model = ssdlite320_mobilenet_v3_large(pretrained=True)
    # model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    # model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    # model = retinanet_resnet50_fpn(pretrained=True)
    # model = maskrcnn_resnet50_fpn(pretrained=True)
    # model = ssd300_vgg16(pretrained=True)
    model = model.eval()
    model.cuda()

    # COCO Datasets.
    coco = COCO(annotation_file=args.annotation_path)
    image_ids = coco.getImgIds()

    num_steps = len(image_ids)
    elapsed_list = []
    coco_detections = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, image_id in enumerate(image_ids):
        coco_img = coco.imgs[image_id]
        image_width = coco_img["width"]
        image_height = coco_img["height"]

        # Load image.
        if args.read_im == "cv":
            # cv2
            im = cv2.imread(os.path.join(args.images_dir, coco_img["file_name"]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.transpose(2, 0, 1)
            im = np.expand_dims(im, axis=0)
            im = torch.from_numpy(im).to(device)

        elif args.read_im == "torch":
            # torch read_image
            im = read_image(os.path.join(args.images_dir, coco_img["file_name"])).to(
                device
            )
            im = im.unsqueeze(0)

        elif args.read_im == "pil":
            # PIL
            im = Image.open(os.path.join(args.images_dir, coco_img["file_name"]))
            im = pil_to_tensor(im).unsqueeze_(0).to(device)

        else:
            assert False, "You have to choose between cv, torch and pil."

        batch = convert_image_dtype(im, dtype=torch.float)

        # inference.
        start.record()
        with torch.no_grad():
            outputs = model(batch)
        end.record()
        torch.cuda.synchronize()

        bboxes = outputs[0]["boxes"].cpu().detach().numpy()
        scores = outputs[0]["scores"].cpu().detach().numpy()
        class_ids = outputs[0]["labels"].cpu().detach().numpy()

        inference_time = start.elapsed_time(end)
        elapsed_list.append(inference_time)

        for index, bbox in enumerate(bboxes):
            score = scores[index]
            x1 = int(bbox[0])
            x2 = int(bbox[2])
            y1 = int(bbox[1])
            y2 = int(bbox[3])
            class_id = class_ids[index]

            bbox_coco_fmt = [
                x1,  # x0
                y1,  # x1
                (x2 - x1),  # width
                (y2 - y1),  # height
            ]
            coco_detection = {
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": [int(coord) for coord in bbox_coco_fmt],
                "score": float(score),
            }
            # print(coco_detection)
            coco_detections.append(coco_detection)

        if (i + 1) % args.display_every == 0:
            print(
                "  step %03d/%03d, iter_time(ms)=%.0f"
                % (i + 1, num_steps, elapsed_list[-1])
            )

    # write coco detections to file
    coco_detections_path = os.path.join(".", "coco_detections.json")
    with open(coco_detections_path, "w") as f:
        json.dump(coco_detections, f)

    cocoDt = coco.loadRes(coco_detections_path)

    # compute coco metrics
    eval = COCOeval(coco, cocoDt, "bbox")
    eval.params.imgIds = image_ids

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    os.remove(coco_detections_path)

    output = []
    output.append(args.model_name)
    output.extend(eval.stats)
    with open(args.output, "a") as f:
        writer = csv.writer(f)
        writer.writerow(output)

    results = {}
    iter_times = np.array(elapsed_list)
    results["total_time"] = np.sum(iter_times)
    iter_times = iter_times[args.num_warmup_iterations :]
    results["images_per_sec"] = np.mean(1 / iter_times * 1000)
    results["99th_percentile"] = np.percentile(iter_times, q=99, interpolation="lower")
    results["latency_mean"] = np.mean(iter_times)
    results["latency_median"] = np.median(iter_times)
    results["latency_min"] = np.min(iter_times)

    print("  images/sec: %d" % results["images_per_sec"])
    print("  99th_percentile(ms): %.2f" % results["99th_percentile"])
    print("  total_time(s): %.1f" % results["total_time"])
    print("  latency_mean(ms): %.2f" % results["latency_mean"])
    print("  latency_median(ms): %.2f" % results["latency_median"])
    print("  latency_min(ms): %.2f" % results["latency_min"])


if __name__ == "__main__":
    main()
