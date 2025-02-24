## Installation Guide for NVIDIA Jetson AGX Orin Developer Kit

### PyTorch installation

```bash
# important!:
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

pip install --no-cache $TORCH_INSTALL

pip install torch-sparse==0.6.17

pip install torch-scatter==2.1.1

```


### Torchvision installation

Follow instructions in https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 -> Installation -> Torchvision. Ensure the compatible torchvision version is selected. In this case, v0.15.1.

Install it from source.