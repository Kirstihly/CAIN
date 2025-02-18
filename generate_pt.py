import torch
import mtk_converter

from model.cain import CAIN

torch.cuda.empty_cache()

print("Building model: CAIN")
model = CAIN(depth=3)
model = torch.nn.DataParallel(model).to("cpu")

checkpoint = torch.load("pretrained_cain.pth")
model.load_state_dict(checkpoint["state_dict"], strict=True)

trace_data = torch.randn((1, 3, 768, 1536))
trace_model = torch.jit.trace(model.cpu().eval(), (trace_data, trace_data))
torch.jit.save(trace_model, "cain.pt")

converter = mtk_converter.PyTorchConverter.from_script_module_file(
    "cain.pt", [[1, 3, 768, 1536], [1, 3, 768, 1536]]
)
# Convert to TFLite format
_ = converter.convert_to_tflite(output_file="cain.tflite")
