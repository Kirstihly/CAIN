# PyTorch-ONNX-TFLite
# https://github.com/sithu31296/PyTorch-ONNX-TFLite
import torch
import onnx

# import tensorflow as tf

# from onnx_tf.backend import prepare
from model.cain import CAIN

sample_input = torch.rand((1, 3, 768, 1536), device="cpu")

print("#################################")
print("Building ONNX model: CAIN")
model = CAIN(depth=3)
model = torch.nn.DataParallel(model)

checkpoint = torch.load("pretrained_cain.pth", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.eval()

torch.onnx.export(
    model.module.cpu(),  # PyTorch Model
    (sample_input, sample_input),  # Input tensor
    "tmp_cain.onnx",  # Output file (eg. 'output_model.onnx')
    opset_version=12,  # Operator support version
    input_names=["frame_1", "frame_2"],  # Input tensor name (arbitary)
    output_names=["frame_interp"],  # Output tensor name (arbitary)
    do_constant_folding=False,
)

print("#################################")
print("Verification and Inference")
model = onnx.load("tmp_cain.onnx")

onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)

# print("#################################")
# print("Building TF model: CAIN")
# onnx_model = onnx.load("tmp_cain.onnx")
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph("tf_model_path")

# tf_model = tf.saved_model.load("tf_model_path")
# tf_model.trainable = False

# out = tf_model(**{"frame_1": sample_input, "frame_2": sample_input})
