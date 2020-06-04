import torchvision
import time
import copy
import torch
import os
import multiprocessing
import dataloader

def new_model(chosen_model, pretrained, num_classes):
    """Loads the chosen model.
    
    Args:
        chosen_model    (str): The name of the chosen model.
        pretrained      (bool): True for pretrained model, False otherwise.
        num_classes     (int): The number of classes of the dataset.
    
    Returns:
        The chosen model if chosen_model is valid, None otherwise.
    """
    if chosen_model == "alexnet":
        model = torchvision.models.alexnet(pretrained=pretrained, progress=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif chosen_model == "vgg11":
        model = torchvision.models.vgg11(pretrained=pretrained, progress=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif chosen_model == "vgg11_bn":
        model = torchvision.models.vgg11_bn(pretrained=pretrained, progress=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif chosen_model == "vgg13":
        model = torchvision.models.vgg13(pretrained=pretrained, progress=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif chosen_model == "vgg13_bn":
        model = torchvision.models.vgg13_bn(pretrained=pretrained, progress=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif chosen_model == "vgg16":
        model = torchvision.models.vgg16(pretrained=pretrained, progress=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif chosen_model == "vgg16_bn":
        model = torchvision.models.vgg16_bn(pretrained=pretrained, progress=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif chosen_model == "vgg19":
        model = torchvision.models.vgg19(pretrained=pretrained, progress=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif chosen_model == "vgg19_bn":
        model = torchvision.models.vgg19_bn(pretrained=pretrained, progress=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif chosen_model == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained, progress=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    elif chosen_model == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained, progress=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    elif chosen_model == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained, progress=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    elif chosen_model == "resnet101":
        model = torchvision.models.resnet101(pretrained=pretrained, progress=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    elif chosen_model == "resnet152":
        model = torchvision.models.resnet152(pretrained=pretrained, progress=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    elif chosen_model == "squeezenet1_0":
        model = torchvision.models.squeezenet1_0(pretrained=pretrained, progress=True)
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    elif chosen_model == "squeezenet1_1":
        model = torchvision.models.squeezenet1_1(pretrained=pretrained, progress=True)
        model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    elif chosen_model == "densenet121":
        model = torchvision.models.densenet121(pretrained=pretrained, progress=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    elif chosen_model == "densenet161":
        model = torchvision.models.densenet161(pretrained=pretrained, progress=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    elif chosen_model == "densenet169":
        model = torchvision.models.densenet169(pretrained=pretrained, progress=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    elif chosen_model == "densenet201":
        model = torchvision.models.densenet201(pretrained=pretrained, progress=True)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    elif chosen_model == "inception_v3":
        model = torchvision.models.inception_v3(pretrained=pretrained, progress=True)
        model.AuxLogits.fc = torch.nn.Linear(768, num_classes)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    elif chosen_model == "googlenet":
        model = torchvision.models.googlenet(pretrained=pretrained, progress=True)
    elif chosen_model == "shufflenet_v2_x0_5":
        model = torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained, progress=True)
    elif chosen_model == "shufflenet_v2_x1_0":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained, progress=True)
    elif chosen_model == "shufflenet_v2_x1_5":
        model = torchvision.models.shufflenet_v2_x1_5(pretrained=pretrained, progress=True)
    elif chosen_model == "shufflenet_v2_x2_0":
        model = torchvision.models.shufflenet_v2_x2_0(pretrained=pretrained, progress=True)
    elif chosen_model == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(pretrained=pretrained, progress=True)
    elif chosen_model == "resnext101_32x8d":
        model = torchvision.models.resnext101_32x8d(pretrained=pretrained, progress=True)
    elif chosen_model == "wide_resnet50_2":
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained, progress=True)
    elif chosen_model == "wide_resnet101_2":
        model = torchvision.models.wide_resnet101_2(pretrained=pretrained, progress=True)
    elif chosen_model == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(pretrained=pretrained, progress=True)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif chosen_model == "ResNeXt-101-32x8d":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    else:
        print("Chosen model name not yet accepted by this code.")
        return None
    return model

def get_model(model_name, pretrained, load_choice, num_classes, models_path):
    """Loads the chosen model based on load_choice. 
        If load_choice < 0, a new model is loaded. 
        Otherwise, models_path is searched to find a saved model.
        This function assumes that models are saved as models_path/MODEL_NAME/EPOCH_NUM.pth
    
    Args:
        model_name      (str): The name of the chosen model.
        pretrained      (bool): True for pretrained model, False otherwise.
        load_choice     (int): The model version to be loaded. -1 for a new model.
        num_classes     (int): The number of classes of the dataset.
        models_path      (str): The folder path where saved models are stored.
    
    Returns:
        A saved model if load_choice >= 0. If load_choice < 0, 
            a pretrained model is loaded if pretrained=True and 
            an untrained model if pretrained=False.
        The previous epoch of the loaded model.
    """
    if load_choice >= 0:
        if os.path.isdir(models_path + model_name):
            filename = f"{models_path + model_name}/{load_choice}.pth"
            if os.path.isfile(filename):
                return torch.load(filename), load_choice
            else:
                print(f"{filename} not found!")
                return None, -1
        else:
            print(f"No folder {model_name} found in {models_path}")
            return None, -1
    else:
        return new_model(model_name, pretrained, num_classes), -1

def train_model(model, model_name, models_path, model_epoch, dataloaders, loss_function, optimizer, num_epochs, device, is_inception=False):
    """Trains the model and saves a copy along with the loss and accuracy every epoch.
    
    Args:
        model           (models): The model to be trained.
        model_name      (str): The name of the model.
        models_path      (str): The folder path where saved models are stored.
        model_epoch     (int): The last epoch of the loaded model.
        dataloaders     (dict): Dictionary of {train:training dataloader, val:validation dataloader}
        loss_function   (loss): The loss function.
        optimizer       (optim): The optimizer.
        num_epochs      (int): The number of epochs to train for.
        device          (device): The GPU device to train on.
        is_inception    (bool): True if the model is the Inception network.
    """
    for epoch in range(num_epochs+1):
        epoch_time = time.perf_counter()
        if model_epoch >= epoch:
            continue
        print(f"Epoch {epoch}/{num_epochs}")
        print('-' * 10)
        epoch_loss_acc = ""
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            phase_time = time.perf_counter()

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # data_time = time.perf_counter()

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # data_time_elapsed = time.perf_counter() - data_time
                # print('Time taken for data loading: {}'.format(data_time_elapsed))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # forward_backward_time = time.perf_counter()

                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = loss_function(outputs, labels)
                        loss2 = loss_function(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                        del loss1
                        del loss2
                    else:
                        if phase == 'train':
                            outputs = model(inputs)
                        else:
                            with torch.no_grad():
                                outputs = model(inputs)
                        loss = loss_function(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    del outputs

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # forward_backward_time_elapsed = time.perf_counter() - forward_backward_time
                    # print('Time taken for forward backward: {}'.format(forward_backward_time_elapsed))

                # running_loss_time = time.perf_counter()

                # statistics
                running_loss += float(loss) * inputs.size(0)
                running_corrects += float(torch.sum(preds == labels.data))
                del preds

                # running_loss_time_elapsed = time.perf_counter() - running_loss_time
                # print('Time taken for loss calculation: {}'.format(running_loss_time_elapsed))

                # data_time = time.perf_counter()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            del running_loss
            del running_corrects

            loss_msg = '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)
            print(loss_msg)
            epoch_loss_acc += str(loss_msg + "\n")
            
            phase_time_elapsed = time.perf_counter() - phase_time
            print('Time taken for {}: {}'.format(phase, phase_time_elapsed))

        print()
        if not os.path.isdir(models_path + model_name + '/'):
            os.mkdir(models_path + model_name + '/')
        torch.save(model, f"{models_path + model_name + '/' + str(epoch)}.pth")
        with open(f"{models_path + model_name + '/' + str(epoch)}.txt","w") as outfile:
            outfile.write(epoch_loss_acc)
        epoch_time_elapsed = time.perf_counter() - epoch_time
        print('Time taken for this epoch: {}'.format(epoch_time_elapsed))

def train_driver(model_name, pretrained, load_choice,
                    models_path, train_data_path, val_data_path, num_epochs,
                    batch_size, image_size, num_workers, use_num_worker_mult=True):
    """Loads and trains the selected model.
    
    Attributes:
        model                   (models): The model to be trained.
        model_epoch             (int): The last epoch of the loaded model.
        dataloaders             (dict): Dictionary of {train:training dataloader, val:validation dataloader}
        loss_function           (loss): The loss function.
        optimizer               (optim): The optimizer.
        is_inception            (bool): True if the model is the Inception network.
        device                  (device): The GPU device to train on.

    Args:
        model_name              (str): The name of the model.
        pretrained              (bool): True to load a pretrained model, False to load an untrained model.
        load_choice             (int): The model version to be loaded. -1 for a new model.
        models_path              (str): The folder path where saved models are stored.
        train_data_path         (str): The folder path where training images are stored.
        val_data_path           (str): The folder path where validation images are stored.
        num_epochs              (int): The number of epochs to train for.
        batch_size              (int): The number of images per batch.
        image_size              (int): The size for all images to be resized to.
        num_workers             (int): The number of workers for DataLoader.
        use_num_worker_mult     (bool): If False, number of workers = num_workers. 
                                            If True, number of workers = num_workers times number of CPU cores.
    """
    if use_num_worker_mult:
        cpu_cores = int(multiprocessing.cpu_count())
        num_workers = num_workers * cpu_cores
        print(f"Number of CPU cores: {cpu_cores}")
    print(f"Number of workers: {num_workers}")

    train_loader, val_loader, num_classes = dataloader.get_train_data(train_data_path, 
                                                    val_data_path, 
                                                    batch_size=batch_size, 
                                                    image_size=image_size,
                                                    num_workers=num_workers)
    dataloaders = {'train':train_loader, 'val':val_loader}

    model, model_epoch = get_model(model_name, pretrained, load_choice, num_classes, models_path)
    print(f"Previous epoch: {model_epoch}")

    if model is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device != "cpu":
            if torch.cuda.device_count()>1:
                print(f"{torch.cuda.device_count()}GPUs detected.")
                model = torch.nn.DataParallel(model)
        model = model.to(device)
        print(f"Device: {device}")

        loss_function = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train_model(model, model_name, models_path, model_epoch, dataloaders, 
                    loss_function, optimizer, num_epochs, device,
                    is_inception=(model_name=="inception"))