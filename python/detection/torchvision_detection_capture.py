#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TensorRT Object detection.
    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import colorsys
import os
import random
import time

import cv2
import numpy as np

import torch
import torchvision

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks

WINDOW_NAME = "PyTorch detection example."

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def read_label_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


def random_colors(N):
    N = N + 1
    hsv = [(i / N, 1.0, 1.0) for i in range(N)]
    colors = list(
        map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv)
    )
    random.shuffle(colors)
    return colors


def draw_rectangle(image, box, color, thickness=3):
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness)


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label", help="File path of label file.", default=None, type=str
    )
    parser.add_argument(
        "--videopath", help="File path of input video file.", default=None, type=str
    )
    parser.add_argument(
        "--output", help="File path of output vide file.", default=None, type=str
    )
    parser.add_argument(
        "--score_threshold", help="Score threshold.", default=0.5, type=float
    )
    parser.add_argument(
        "--proba_threshold", help="Proba threshold.", default=0.5, type=float
    )
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # Read label and generate random colors.
    label_strings = read_label_file(args.label) if args.label else None
    last_key = sorted(label_strings.keys())[len(label_strings.keys()) - 1]
    random.seed(42)
    colors = random_colors(last_key)

    # Video capture.
    if args.videopath == "":
        print("open camera.")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        print("open video file", args.videopath)
        cap = cv2.VideoCapture(args.videopath)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Input Video (height, width, fps): ", h, w, fps)

    # Load model.
    # model = ssdlite320_mobilenet_v3_large(pretrained=True)
    # model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model = model.eval()
    model.cuda()

    # Output Video file
    # Define the codec and create VideoWriter object
    video_writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    elapsed_list = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = im.transpose(2, 0, 1)
        im = np.expand_dims(im, axis=0)
        im = torch.from_numpy(im).cuda()
        batch = convert_image_dtype(im, dtype=torch.float)

        # inference.
        start.record()
        with torch.no_grad():
            outputs = model(batch)

        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end)

        score_threshold = args.score_threshold
        proba_threshold = args.proba_threshold

        bboxes = outputs[0]["boxes"][outputs[0]["scores"] > score_threshold]
        scores = outputs[0]["scores"][outputs[0]["scores"] > score_threshold]
        labels = outputs[0]["labels"][outputs[0]["scores"] > score_threshold]
        boolean_masks = (
            outputs[0]["masks"][outputs[0]["scores"] > score_threshold]
            > proba_threshold
        )

        bboxes = bboxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        boolean_masks = boolean_masks.squeeze(1).cpu().detach().numpy()

        # Draw mask.
        display_im = frame
        mask_im = frame
        for index, bbox in enumerate(bboxes):
            label_id = labels[index]

            boolean_mask = np.where(boolean_masks[index] == True, 255, 0).astype(
                np.uint8
            )
            boolean_mask = cv2.cvtColor(boolean_mask, cv2.COLOR_GRAY2BGR)
            mask_color = np.full(display_im.shape, colors[label_id], np.uint8)
            dst = cv2.bitwise_and(mask_color, boolean_mask)
            mask_im = cv2.bitwise_or(mask_im, dst)

        display_im = cv2.addWeighted(display_im, 0.3, mask_im, 1 - 0.3, 0)
        for index, bbox in enumerate(bboxes):
            label_id = labels[index]

            # Draw bounding box.
            caption = "{0}({1:.2f})".format(label_strings[label_id], scores[index])

            xmin = int(bbox[0])
            xmax = int(bbox[2])
            ymin = int(bbox[1])
            ymax = int(bbox[3])
            draw_rectangle(display_im, (xmin, ymin, xmax, ymax), colors[label_id])
            draw_caption(display_im, (xmin, ymin - 10), caption)

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = "Mask R-CNN ResNet-50 FPN" + " " + fps_text + avg_text
        draw_caption(display_im, (10, 30), display_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(display_im)

        # Display
        cv2.imshow(WINDOW_NAME, display_im)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    # When everything done, release the window
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
