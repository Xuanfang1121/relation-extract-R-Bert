# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 20:15
# @Author  : zxf
import onnx
from onnx import defs
from onnx_tf.backend import prepare

print(defs.has("DynamicSlice"))
model_onnx = onnx.load('./model/model_simple.onnx')
# Check the model
# onnx.checker.check_model(model_onnx)

tf_rep = prepare(model_onnx, strict=False)

# Export model as .pb file
tf_rep.export_graph('./model/model_simple.pb')