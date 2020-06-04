import torchvision
import imgaug
import numpy as np
import PIL
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

def get_train_data(train_data_path, val_data_path, 
             batch_size=8, image_size=256,
             num_workers=1):
    """Loads the image data from the specified training and validation folders, and applies image augmentation to them.

    Args:
        train_data_path     (str): The folder path that contains training images.
        val_data_path       (str): The folder path that contains validation images.
        batch_size          (int): The number of images per batch.
        image_size          (int): The size for all images to be resized to.
        num_workers         (int): The number of workers for DataLoader.

    Returns:
        train_loader        (DataLoader): Dataloader containing training images.
        val_loader          (DataLoader): Dataloader containing validation images.
        num_classes         (int): The number of classes detected from the training folder.
    """
    class ImgAugTransform:
        """An imgaug transformation class that applies Gaussian Blur to images."""
        def __init__(self):
            self.aug = imgaug.augmenters.Sequential([
                imgaug.augmenters.Sometimes(0.25, imgaug.augmenters.GaussianBlur(sigma=(0, 3.0)))
            ])
            
        def __call__(self, img):
            img = np.array(img)
            return self.aug.augment_image(img)

    train_augmentations = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(179.9, expand=True, fill=255),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.25, hue=0.5),
        ImgAugTransform(),
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((image_size,image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_augmentations = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size,image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_data_path, transform=train_augmentations)
    val_dataset = torchvision.datasets.ImageFolder(val_data_path, transform=val_augmentations)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=num_workers,
                                            drop_last=True)
    num_classes = len(train_dataset.classes)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=num_workers,
                                            drop_last=True)

    return train_loader, val_loader, num_classes

def get_test_data(test_data_path, batch_size=8, image_size=256, num_workers=1):
    """Loads the image data from the specified training and validation folders, and applies image augmentation to them.

    Args:
        test_data_path     (str): The folder path that contains testing images.
        batch_size         (int): The batch size of DataLoader.
        image_size         (int): The size for all images to be resized to.
        num_workers        (int): The number of workers for DataLoader.

    Returns:
        test_loader        (DataLoader): Dataloader containing testing images.
        num_classes        (int): The number of classes detected from the testing folder.
    """

    test_augmentations = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size,image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = torchvision.datasets.ImageFolder(test_data_path, transform=test_augmentations)

    # class_ids = dict((v,k) for k,v in dict(enumerate(dataset.classes)).items())

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=num_workers,
                                            drop_last=True)
    num_classes = len(test_dataset.classes)

    return test_loader, num_classes
