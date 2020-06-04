import os
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_array(x_array, y_array, x_label, y_label, is_accuracy):
    """Draws a plot based on the input X Y arrays and labels.
        If the plot type is accuracy, the Y axis is 0 to 1, since 1 is the max(1.00 = 100%).
        If the plot type is not accuracy, then it should be loss. The Y is will be 0 to max+1."""
    plt.plot(x_array, y_array)
    plt.xticks(np.arange(0, max(x_array)+2, 1))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if is_accuracy:
        plt.axis([0, max(x_array)+1, 0, 1])
    else:
        plt.axis([0, max(x_array)+1, 0, math.ceil(max(y_array))])
    plt.show()

def plot_history(model_name, models_path):
    """Reads all the accuracy .txt files and plots training loss, training accuracy, validation loss and validation accuracy against epochs."""
    model_folder = models_path + model_name
    if os.path.isdir(model_folder):
        num_models = sum([sum(name.endswith(".txt") for name in files) for path, subdirs, files in os.walk(model_folder)])
        history_epochs = [0 for _ in range(num_models)]
        train_losses = [0 for _ in range(num_models)]
        train_accuracies = [0 for _ in range(num_models)]
        val_losses = [0 for _ in range(num_models)]
        val_accuracies = [0 for _ in range(num_models)]
        for path, subdirs, files in os.walk(model_folder):
            for name in files:
                if name.endswith(".txt"):
                    index = int(name.split('.')[0])
                    history_epochs[index] = int(name.split('.')[0])
                    with open(path + '/' + name, 'r') as history_file:
                        lines = history_file.readlines()
                        train_losses[index] = float(lines[0].split("train Loss: ")[1].split(" Acc: ")[0])
                        train_accuracies[index] = float(lines[0].split("train Loss: ")[1].split(" Acc: ")[1])
                        val_losses[index] = float(lines[1].split("val Loss: ")[1].split(" Acc: ")[0])
                        val_accuracies[index] = float(lines[1].split("val Loss: ")[1].split(" Acc: ")[1])
        if len(train_losses) > 0:
            print(history_epochs)
            plot_array(history_epochs, train_losses, "Epochs", "Train Loss", False)
            print(train_losses)
            plot_array(history_epochs, train_accuracies, "Epochs", "Train Accuracy", True)
            print(train_accuracies)
            plot_array(history_epochs, val_losses, "Epochs", "Validation Loss", False)
            print(val_losses)
            plot_array(history_epochs, val_accuracies, "Epochs", "Validation Accuracy", True)
            print(val_accuracies)
        else:
            print(f"No history found in {model_folder}")
    else:
        print(f"No folder {model_name} found in {models_path}")