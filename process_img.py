import argparse
import cv2
import glob
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", help="Path to the input dir of pngs", required=True
    )
    parser.add_argument("--output_dir", help="Path to the output dir", required=True)
    parser.add_argument(
        "--input_shapes", help="Specify image shapes of N,C,H,W", default="1,4,540,1216"
    )
    parser.add_argument(
        "--output_dtype",
        help="Specify output data type",
        default="float32",
        choices=("float16", "float32", "float64", "uint8"),
    )
    parser.add_argument(
        "--output_ext",
        help="Specify output file extension",
        default="npy",
        choices=("npy", "raw", "bin"),
    )
    parser.add_argument(
        "--bgr2rgb",
        help="Specify output channel format",
        action="store_true",
    )
    args = parser.parse_args()

    ########################
    # Validate input_dir
    ########################
    pngs = glob.glob(os.path.join(args.input_dir, "*.png"))
    assert len(pngs) > 0, "Error: Input dir is empty!"

    ########################
    # Validate output_dir
    ########################
    os.makedirs(args.output_dir, exist_ok=True)
    assert len(os.listdir(args.output_dir)) == 0, "Error: Output dir is not empty!"

    ########################
    # Validate input_shapes
    ########################
    nhwc = {"n": 0, "c": 0, "h": 0, "w": 0}
    try:
        nhwc["n"], nhwc["c"], nhwc["h"], nhwc["w"] = map(
            int, args.input_shapes.split(",")
        )
    except:
        print("Error: Loading shape failed!")
        exit()
    assert nhwc["n"] == 1, "Error: NHWC wrong. Only supporting N=1"
    assert (
        nhwc["c"] == 3 or nhwc["c"] == 4
    ), "Error: NHWC wrong. Only supporting C=3 or C=4"

    pngs.sort()
    for i in range(len(pngs)):
        filen = i + 1
        im = cv2.imread(pngs[i], cv2.IMREAD_COLOR)
        if args.bgr2rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if args.output_dtype == "float16":
            im = im.astype(np.float16) / 255.0
        elif args.output_dtype == "float32":
            im = im.astype(np.float32) / 255.0
        elif args.output_dtype == "float64":
            im = im.astype(np.float64) / 255.0
        elif args.output_dtype == "uint8":
            im = im.astype(np.uint8)
        else:
            raise Exception("Error: Data type not supported!")

        width, height, channel = im.shape
        if width != nhwc["w"] or height != nhwc["h"]:
            im = cv2.resize(im, (nhwc["w"], nhwc["h"]))

        if nhwc["c"] == 4:
            if args.output_dtype == "uint8":
                im = np.dstack((im, 255 * np.ones((nhwc["h"], nhwc["w"]))))
            else:
                im = np.dstack((im, np.ones((nhwc["h"], nhwc["w"]))))

        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 1, 2))

        # Enforce data type again because resize will change the type
        if args.output_dtype == "float16":
            im = im.astype(np.float16)
        elif args.output_dtype == "float32":
            im = im.astype(np.float32)
        elif args.output_dtype == "float64":
            im = im.astype(np.float64)
        elif args.output_dtype == "uint8":
            im = im.astype(np.uint8)

        if args.output_ext == "npy":
            np.save(os.path.join(args.output_dir, "batch_" + str(filen) + ".npy"), im)
        elif args.output_ext == "raw":
            im.tofile(os.path.join(args.output_dir, "batch_" + str(filen) + ".raw"))
        elif args.output_ext == "bin":
            im.tofile(os.path.join(args.output_dir, "batch_" + str(filen) + ".bin"))
        else:
            raise Exception("Error: File extension not supported!")