# Retail-Object-Recognition
The primary purpose of this repository is to train multiple different models on the [RPC(Retail Product Checkout)](https://rpc-dataset.github.io/) dataset.\
The intention is to allow you to compare the Testing results of each model after training on the RPC dataset.\
Interactive Google Colab Version: [ROR](https://colab.research.google.com/drive/1LS9oUxVPts0rza-HT5aAGbUCATTnn9Mh?usp=sharing)

## Repository Features 
  Crop images from the RPC dataset.\
  Verify cropped images.\
  Train models from Torchvision on the cropped images.\
  Test trained models.\
  Plot training loss, training accuracy, testing loss and testing accuracy from trained models.

## Prerequisites
  ### Before running this repository, you need the following:
    Python >= 3.5
    OpenCV
    PyTorch
    Torchvision
    Pillow
    ImgAug
    PyYaml
    
## Usage:
  Input `python main.py CONFIG_FILE.yml`\
  A config guide as well as sample configs have been provided in `configs/`.\
  The config folder can be changed in `main.py`
