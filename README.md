# SFNet
This project is the PyTorch implementation of the paper 'Dual-Domain Perception and Cross-Domain Collaboration Guidance Network for Image Inpainting'.
# Dependencies
We use python to build our code. Please make sure the following dependencies are installed:
* **Python** 3.7
* **PyTorch** 1.13.1
* **CUDA** 11.6 + **cuDNN** 8.5
* **TorchVision** 0.4.1
> Training and inference require an **NVIDIA GPU**.

# Download Datasets

We use the following datasets in our experiments:

* **[Places2](http://places2.csail.mit.edu/)**
* **[CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)**
* **[Paris StreetView](https://github.com/pathak22/context-encoder)**

For testing, we adopt **[Irregular Masks (12k)](https://github.com/lamhoangtung/MaskDataset)** provided by *Liu et al.*

