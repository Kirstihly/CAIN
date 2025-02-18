#
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import os
import tensorflow as tf
import numpy as np


if "QNN_SDK_ROOT" not in os.environ:
    raise RuntimeError("QNN_SDK_ROOT not setup.  Please run the SDK env setup script.")

global QNN_ROOT
QNN_ROOT = os.path.abspath(os.environ["QNN_SDK_ROOT"])
CWD = os.getcwd()

converter = tf.lite.TFLiteConverter.from_saved_model("tf_model_path")


def representative_dataset():
    with open(CWD + "/frame_seq_raw/target_raw_list.txt", "r") as file:
        for line in file:
            line = line.strip()
            frame_1, frame_2 = line.split(" ")
            frame_1 = CWD + "/frame_seq_raw_f32/" + frame_1
            frame_2 = CWD + "/frame_seq_raw_f32/" + frame_2
            raw_image1 = np.fromfile(frame_1, dtype = np.float32, count = 3*768*1536).reshape(1, 3, 768, 1536)
            raw_image2 = np.fromfile(frame_2, dtype = np.float32, count = 3*768*1536).reshape(1, 3, 768, 1536)
            yield [raw_image1, raw_image2]


converter.experimental_new_converter = True  # True for MLIR / False for TOCO
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_type = tf.uint8  # tf.uint8 or tf.int8
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    # tf.lite.OpsSet.SELECT_TF_OPS,
]

tflite_model = converter.convert()
# Save the model.
with open("supported_ops_cain.tflite", "wb") as f:
    f.write(tflite_model)
