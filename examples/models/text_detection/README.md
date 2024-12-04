```sh
CRATE_NAME=Triton-TensorRT-Inference-CRAFT-pytorch

git clone https://github.com/k9ele7en/Triton-TensorRT-Inference-CRAFT-pytorch && cd $CRATE_NAME

# mkdir model_repository/detec_onnx/1

pip install -r requirements.txt

# download [craft_mlt_25k.pth](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)

mv ./craft_mlt_25k.pth weights/

mkdir ../model_repository/detec_onnx/1

cd converters

rm line 7 and 25 from basenet vgg16bn.py
python3 pth2onnx.py

cd ../..

mv $CRATE_NAME/model_repository/detec_onnx/1/detec_onnx.onnx craft/1/model.onnx

rm -rf $CRATE_NAME
```