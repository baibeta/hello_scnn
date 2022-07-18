#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-07-18 10:09
import os
import cv2
import json
import config
import numpy as np
import uuid
import pickle
import shutil
from collections import namedtuple


def generate_label(flist, mode):
    output = []
    for x in flist:
        x = os.path.join(config.DATA_PATH, x)

        with open(x) as f:
            print(x)
            for line in f:
                label = json.loads(line)
                y_points = label["h_samples"]
                left_lanes = []
                right_lanes = []
                for i, x_points in enumerate(label["lanes"]):
                    points = [(x, y) for (x, y) in zip(x_points, y_points) if x > 0]
                    if len(points) < 4:
                        continue
                    slope = (
                        np.arctan2(
                            points[-1][1] - points[0][1], points[-1][0] - points[0][0]
                        )
                        / np.pi
                        * 180
                    )

                    if slope <= 90:
                        left_lanes.append((points, slope))
                    else:
                        right_lanes.append((points, slope))

                left_lanes.sort(key=lambda x: x[1])
                right_lanes.sort(key=lambda x: x[1])
                left_lanes = left_lanes[-2:]
                right_lanes = right_lanes[:2]

                image = label["raw_file"]
                label_image = np.zeros((config.ORIG_IMAGE_H, config.ORIG_IMAGE_W, 3))
                exist = []

                lanes = []
                for i in range(2 - len(left_lanes)):
                    lanes.append(None)
                    exist.append(0)
                for p, _ in left_lanes + right_lanes:
                    lanes.append(p)
                    exist.append(1)
                for i in range(2 - len(right_lanes)):
                    lanes.append(None)
                    exist.append(0)

                assert len(lanes) == 4

                for i, p in enumerate(lanes):
                    if p is None:
                        continue
                    for pa, pb in zip(p[:-1], p[1:]):
                        cv2.line(
                            label_image,
                            tuple(pa),
                            tuple(pb),
                            (i + 1, 0, 0),
                            config.SEG_WIDTH,
                        )

                label_image_file = os.path.join(
                    label_image_dir, f"{str(uuid.uuid1())}.png"
                )
                cv2.imwrite(
                    os.path.join(config.DATA_PATH, label_image_file),
                    label_image,
                )
                output.append((image, label_image_file, exist))

    with open(os.path.join(config.DATA_PATH, "label", f"{mode}.dat"), "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    label_image_dir = os.path.join("label", "clips")
    shutil.rmtree(os.path.join(config.DATA_PATH, label_image_dir), ignore_errors=True)
    os.makedirs(os.path.join(config.DATA_PATH, label_image_dir), exist_ok=True)

    generate_label(
        ("label_data_0313.json", "label_data_0531.json", "label_data_0601.json"),
        "train",
    )

    generate_label(
        ("test_label.json",),
        "test",
    )
