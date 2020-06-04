import os
import torch
import multiprocessing
import dataloader

def get_test_model(model_name, load_choice, models_path):
    """Loads the chosen model based on load_choice. 
        If load_choice < 0, the accuracy files of every version of the model is compared and 
            the highest validation accuracy versionn is loaded.
        Otherwise, the specified model is loaded.
        This function assumes that models are saved as models_path/MODEL_NAME/EPOCH_NUM.pth
    
    Args:
        model_name      (str): The name of the chosen model.
        load_choice     (int): The model version to be loaded. -1 for the highest validation accuracy model.
        models_path      (str): The folder path where saved models are stored.
    
    Returns:
        The specified model if load_choice >= 0, the highest validation accuracy model otherwise.
        The name of the loaded model.
    """
    model_folder = models_path + model_name
    if os.path.isdir(model_folder):
        if load_choice < 0:
            val_accuracies = {}
            for path, subdirs, files in os.walk(model_folder):
                for name in files:
                    if name.endswith(".txt"):
                        with open(path + '/' + name, 'r') as history_file:
                            lines = history_file.readlines()
                            val_accuracies[name.split('.')[0]] = float(lines[1].split("val Loss: ")[1].split(" Acc: ")[1])
            if len(val_accuracies) > 0:
                load_choice = int(max(val_accuracies, key=lambda key: val_accuracies[key]))
                print(f"Highest Validation model: {load_choice}")
        try:
            loaded_model = torch.load(f"{model_folder}/{load_choice}.pth")
            print(f"Loaded model: {load_choice}.pth")
        except:
            print(f"No {load_choice}.pth found in {model_folder}")

        return loaded_model, f"{load_choice}.pth"
    else:
        print(f"No folder {model_name} found in {models_path}")

def test_model(model, model_name, model_file_name, models_path, test_loader, num_classes, device):
    """Tests the model and outputs the overall accuracy and the accuracy for each class.
    
    Args:
        model           (models): The model to be trained.
        model_name      (str): The name of the model.
        model_name      (str): The filename of the loaded model, which should be {int}.pth.
        models_path      (str): The folder path where saved models are stored.
        test_loader     (DataLoader): The dataloader containing testing images.
        num_classes     (int): The number of classes in the dataset.
        device          (device): The GPU device to train on.
    """
    model.eval()

    class_correct = [0 for _ in range(num_classes+1)]
    class_predicts = [0 for _ in range(num_classes+1)]

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
           # obtain the outputs from the model
            outputs = model.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(labels.size(0)):
                class_predicts[predicted[i]] += 1
                if labels.data[i] == predicted[i]:
                    class_correct[predicted[i]] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    for i in range(num_classes+1):
        print_acc = 0
        if class_predicts[i] > 0:
            print_acc = class_correct[i]/class_predicts[i]
        print(f"Class {i} Accuracy: {print_acc}")

    print('-'*10)
    print(f"Overall Accuracy: {100 * correct / total}%")

def test_driver(model_name, load_choice, models_path, test_data_path, batch_size, image_size, num_workers, use_num_worker_mult=True):
    """Loads and tests the selected model.
    
    Attributes:
        model                   (models): The model to be trained.
        test_loader             (DataLoader): The dataloader containing testing images.
        device                  (device): The GPU device to train on.

    Args:
        model_name              (str): The name of the model.
        load_choice            (int): The model version to be loaded. -1 to load the highest validation accuracy model. 
                                        REQUIRES accuracies to be saved in models_path in the format saved by train.train_model()
        models_path              (str): The folder path where saved models are stored.
        test_data_path          (str): The folder path where testing images are stored.
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
    
    test_loader, num_classes = dataloader.get_test_data(test_data_path, batch_size=batch_size, image_size=image_size, num_workers=num_workers)

    model, model_file_name = get_test_model(model_name, load_choice, models_path)

    if model is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device != "cpu":
            if torch.cuda.device_count()>1:
                print(f"{torch.cuda.device_count()}GPUs detected.")
                model = torch.nn.DataParallel(model)
        model = model.to(device)
        print(f"Device: {device}")

        test_model(model, model_name, model_file_name, models_path, test_loader, num_classes, device)