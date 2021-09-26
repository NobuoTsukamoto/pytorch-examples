#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TorchVision Benchmark.

    Copyright (c) 2021 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import argparse
import os
import time
import json

import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    keypointrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
    keypointrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import convert_image_dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", help="Repeat count.", default=1000, type=int)
    parser.add_argument(
        "--display_every",
        type=int,
        default=500,
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
        "--input_shape",
        type=str,
        default="1080,1980",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--input_batch",
        type=int,
        default=1,
        help="Specify an input batch for inference.",
    )
    parser.add_argument("--output", default="output.json")
    args = parser.parse_args()

    input_shape = tuple(map(int, args.input_shape.split(",")))
    model_names = [
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "retinanet_resnet50_fpn",
        "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large",
        "maskrcnn_resnet50_fpn",
        "keypointrcnn_resnet50_fpn",
    ]

    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in model_names:
        print("Model: %s" % model_name)
        model = globals()[model_name](pretrained=True)
        model = model.eval()
        model.to(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        elapsed_list = []

        for i in range(args.count):
            input = np.random.randint(
                0, 256, (args.input_batch, 3, input_shape[0], input_shape[1])
            )
            input = torch.from_numpy(input).to(device)
            batch = convert_image_dtype(input, dtype=torch.float)

            # inference.
            start.record()
            with torch.no_grad():
                model(batch)
            end.record()
            torch.cuda.synchronize()

            inference_time = start.elapsed_time(end)
            elapsed_list.append(inference_time)

            if (i + 1) % args.display_every == 0:
                print(
                    "  step %03d/%03d, iter_time(ms)=%.0f"
                    % (i + 1, args.count, elapsed_list[-1])
                )
        model = None

        results[model_name] = {}
        iter_times = np.array(elapsed_list)
        results[model_name]["total_time"] = np.sum(iter_times)
        iter_times = iter_times[args.num_warmup_iterations :]
        results[model_name]["images_per_sec"] = np.mean(1 / iter_times * 1000)
        results[model_name]["99th_percentile"] = np.percentile(
            iter_times, q=99, interpolation="lower"
        )
        results[model_name]["latency_mean"] = np.mean(iter_times)
        results[model_name]["latency_median"] = np.median(iter_times)
        results[model_name]["latency_min"] = np.min(iter_times)

        print("  images/sec: %d" % results[model_name]["images_per_sec"])
        print("  99th_percentile(ms): %.2f" % results[model_name]["99th_percentile"])
        print("  total_time(s): %.1f" % results[model_name]["total_time"])
        print("  latency_mean(ms): %.2f" % results[model_name]["latency_mean"])
        print("  latency_median(ms): %.2f" % results[model_name]["latency_median"])
        print("  latency_min(ms): %.2f" % results[model_name]["latency_min"])

    with open(args.output, "w") as file:
        json.dump(results, file, indent=4, separators=(',  ', ':  '))


if __name__ == "__main__":
    main()
