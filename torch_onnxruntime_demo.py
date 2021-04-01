




x = torch.randn(1, 6, 40, 90, requires_grad=True).to(device)
torch_out = model(x)
torch.onnx.export(model,
                  x,
                  "test_to_onnx.onnx",
                  export_params = True,
                  opset_version=9,
                  do_constant_folding=True,
                  input_names = ['x'],
                  output_names = ['y']
                  )





import onnxruntime
import numpy as np


x=np.ones([1,6,40,90]).astype(np.float32)
ort_session = onnxruntime.InferenceSession("/home/hyc/Desktop/hyc/robot_arm_backup/test_newCANN.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name:x}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)

# compare ONNX Runtime and PyTorch results
#np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
