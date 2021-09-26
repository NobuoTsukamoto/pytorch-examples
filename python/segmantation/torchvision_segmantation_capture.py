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
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    fcn_resnet50,
    fcn_resnet101,
    lraspp_mobilenet_v3_large,
)
from torchvision.transforms.functional import convert_image_dtype

WINDOW_NAME = "PyTorch Semantic Segmentation example."


def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def label_to_color_image(colormap, label):
    return colormap[label]


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
        "--videopath", help="File path of input video file.", default=None, type=str
    )
    parser.add_argument(
        "--output", help="File path of output vide file.", default=None, type=str
    )
    parser.add_argument("--score", help="Score threshold.", default=0.2, type=float)
    args = parser.parse_args()

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # Initialize colormap
    random.seed(42)
    colormap = create_pascal_label_colormap()

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
    model_name = "DeepLabV3 ResNet101"
    # model = deeplabv3_mobilenet_v3_large(pretrained=True)
    model = deeplabv3_resnet101(pretrained=True)
    # model = fcn_resnet50(pretrained=True)
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
        # im = cv2.resize(im, (w//2, h//2))
        im = im.transpose(2, 0, 1)
        im = np.expand_dims(im, axis=0)
        im = torch.from_numpy(im).cuda()
        batch = convert_image_dtype(im, dtype=torch.float)

        # inference.
        start.record()
        with torch.no_grad():
            output = model(batch)["out"]
        end.record()
        torch.cuda.synchronize()
        inference_time = start.elapsed_time(end)

        normalized_masks = torch.nn.functional.softmax(output, dim=1)
        class_dim = 0
        all_classes_masks = normalized_masks[0].argmax(class_dim).cpu().detach().numpy()

        seg_image = label_to_color_image(colormap, all_classes_masks)
        # seg_image = cv2.resize(seg_image, (w, h))
        display_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) // 2 + seg_image // 2
        display_im = cv2.cvtColor(display_im, cv2.COLOR_RGB2BGR)

        # Calc fps.
        elapsed_list.append(inference_time)
        avg_text = ""
        if len(elapsed_list) > 100:
            elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list)
            avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

        # Display fps
        fps_text = "Inference: {0:.2f}ms".format(inference_time)
        display_text = model_name + " " + fps_text + avg_text
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
