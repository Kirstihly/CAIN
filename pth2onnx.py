import argparse
import torch
import os
import importlib
import sys


def convert_pth_to_onnx(pth_file, onnx_file, input_size, model):
    """
    Convert a PyTorch .pth weight file to an ONNX format.

    Parameters:
        pth_file (str): Path to the .pth weight file.
        onnx_file (str): Path to save the .onnx file.
        input_size (tuple): Input size for the model (batch_size, channels, height, width).
        model (torch model): PyTorch model loaded with weights.
    """
    if not os.path.exists(pth_file):
        raise FileNotFoundError(f"Weight file {pth_file} does not exist.")

    model.eval()  # Set the model to evaluation mode

    # Prepare dummy input
    dummy_input = torch.randn(*input_size)

    # Export the model to ONNX
    torch.onnx.export(
        model.module,
        (dummy_input, dummy_input),
        onnx_file,
        input_names=["frame_1", "frame_2"],
        output_names=["frame_interp"],
        opset_version=12,
    )

    print(f"Model has been converted to ONNX format and saved at: {onnx_file}")


def loadModel(weight_path, use_cuda):
    from model.cain import CAIN
    model = CAIN(depth=3)
    model = torch.nn.DataParallel(model).to("cpu")

    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    if use_cuda:
        model = model.cuda()
    return model


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pth files to ONNX format."
    )
    parser.add_argument(
        "-p", "--pth", required=True, help="Path to the PyTorch .pth file."
    )
    parser.add_argument(
        "-o", "--onnx", required=True, help="Path to save the ONNX file."
    )
    parser.add_argument(
        "-i",
        "--input_size",
        default="1,3,768,1536",
        help="Input size for the model in the format 'batch,channels,height,width'. Default is '1,3,768,1536'.",
    )
    args = parser.parse_args()

    # Parse the input size argument
    input_size = tuple(map(int, args.input_size.split(",")))

    model = loadModel(args.pth, False)

    # Call the conversion function
    convert_pth_to_onnx(args.pth, args.onnx, input_size, model)
