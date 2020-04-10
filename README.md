## Aimed model
FITVNet(https://arxiv.org/pdf/2001.00346.pdf)

## Baseline codes
Baseline codes comes from (https://github.com/m-tassano/fastdvdnet)
official implementation of FastDVDnet(https://arxiv.org/pdf/1907.01361.pdf)

## User Guide

The code as is runs in Python +3.6 with the following dependencies:

### Dependencies
* [PyTorch v1.0.0](http://pytorch.org/)
* [NVIDIA DALI](https://github.com/NVIDIA/DALI)
* [scikit-image](http://scikit-image.org/)
* [numpy](https://www.numpy.org/)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [tensorboardX](https://github.com/lanpa/tensorboardX/)

Note: the tested version of NVIDIA DALI is 0.10.0. To install it (supposing you have CUDA 10.0), you need to run

```
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali==0.10.0 
```

## Usage

### Testing

If you want to denoise an image sequence using the pretrained model you can execute

```
test_fastdvdnet.py \
	--test_path <path_to_input_sequence> \
	--noise_sigma 30 \
	--save_path results
```

**NOTES**
* The image sequence should be stored under <path_to_input_sequence>
* The model has been trained for values of noise in [5, 55]
* run with *--no_gpu* to run on CPU instead of GPU
* run with *--save_noisy* to save noisy frames
* set *max_num_fr_per_seq* to set the max number of frames to load per sequence
* to denoise _clipped AWGN_ run with *--model_file model_clipped_noise.pth*
* run with *--help* to see details on all input parameters

### Training

If you want to train your own models you can execute

```
train_fastdvdnet.py \
	--trainset_dir <path_to_input_mp4s> \
	--valset_dir <path_to_val_sequences> \
	--log_dir logs
```
