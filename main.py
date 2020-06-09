import sys
import os
import yaml
import config_checker
import image_preprocessor
import train
import test
import plot

is_assume_config_folder = True
config_folder = "configs/"

def make_if_not_exist(filepath):
    """Creates a folder and its subfolders based on filepath."""
    if not os.path.isdir(filepath):
        os.makedirs(filepath)

def driver(yaml_path):
    """Runs the cropping, verifying, training or testing functions based on the configuration provided.

    Args:
        yaml_path   (str): The filepath that contains all the configurations needed.
    """
    if not yaml_path.endswith('.yml'):
        print("Unexpected input detected. Proper usage:\"python main.py CONFIG_NAME.yml\"")
        return

    print(yaml_path)
    config_loaded = False

    if is_assume_config_folder:
        yaml_path = config_folder + yaml_path

    if os.path.isfile(yaml_path):
        with open(yaml_path, 'r') as stream:
            try:
                configs = yaml.safe_load(stream)
                config_loaded = True
            except yaml.YAMLError as exc:
                print(exc)
    else:
        print(f"{yaml_path} not found.")

    if config_loaded:
        is_valid_configs = config_checker.check(configs)
        if is_valid_configs:
            filepaths = configs['filepaths']
            data_types = ['train', 'val', 'test']
            data_paths = filepaths['data_paths']
            cropped_paths = filepaths['cropped_paths']
            instance_paths = filepaths['instances_paths']
            verify_progress_paths = filepaths['verify_progress_paths']

            print(f"data_paths:{data_paths}")
            print(f"cropped_paths:{cropped_paths}")
            print(f"instance_paths:{instance_paths}")
            print(f"verify_progress_paths:{verify_progress_paths}")

            for each in data_types:
                if cropped_paths[each] != None:
                    make_if_not_exist(cropped_paths[each])
            
            if configs['mode'] == "crop":
                selected_type = configs['crop_options']['dataset_choice']
                if selected_type in data_types:
                    image_preprocessor.crop_images_driver(instance_paths[selected_type], data_paths[selected_type], cropped_paths[selected_type])
                    print(f"{selected_type} Image cropping complete.")
                elif selected_type == "all":
                    for each in data_types:
                        image_preprocessor.crop_images_driver(instance_paths[each], data_paths[each], cropped_paths[each])
                        print(f"{each} Image cropping complete.")
                else:
                    print("Unexpected error in configs['crop_options']['dataset_choice'], this should have been caught by config_checker!")

            elif configs['mode'] == "verify":
                selected_type = configs['verify_options']['dataset_choice']
                if selected_type in data_types:
                    image_preprocessor.file_num_compare(instance_paths[selected_type], data_paths[selected_type], cropped_paths[selected_type], selected_type, 
                                                        configs['verify_options']['remove_extra'], configs['verify_options']['crop_missing'])
                elif selected_type == "all":
                    for each in data_types:
                        image_preprocessor.file_num_compare(instance_paths[each], data_paths[each], cropped_paths[each], each, 
                                                            configs['verify_options']['remove_extra'], configs['verify_options']['crop_missing'])
                else:
                    print("Unexpected error in configs['verify_options']['dataset_choice'], this should have been caught by config_checker!")

                if selected_type in data_types:
                    image_preprocessor.verify_jpg_driver(instance_paths[selected_type], data_paths[selected_type],
                        cropped_paths[selected_type], verify_progress_paths[selected_type], 
                        configs['verify_options']['reset_progress'], configs['verify_options']['verify_save_step'])
                    print(f"{selected_type} Verification complete.")
                elif selected_type == "all":
                    for each in data_types:
                        image_preprocessor.verify_jpg_driver(instance_paths[each], data_paths[each],
                                            cropped_paths[each], verify_progress_paths[each], 
                                            configs['verify_options']['reset_progress'], configs['verify_options']['verify_save_step'])
                        print(f"{each} Verification complete.")
                else:
                    print("Unexpected error in configs['verify_options']['dataset_choice'], this should have been caught by config_checker!")

            elif configs['mode'] == "train":
                make_if_not_exist(filepaths['models_path'])
                train_test_options = configs['train_test_options']
                make_if_not_exist(filepaths['models_path'] + train_test_options['model_name'])
                train.train_driver(train_test_options['model_name'], train_test_options['pretrained'], train_test_options['load_choice'],
                                    filepaths['models_path'], cropped_paths['train'], cropped_paths['val'], train_test_options['num_epochs'],
                                    train_test_options['batch_size'], train_test_options['image_size'], 
                                    train_test_options['num_workers'], train_test_options['use_num_worker_mult'])

            elif configs['mode'] == "test":
                train_test_options = configs['train_test_options']
                test.test_driver(train_test_options['model_name'], train_test_options['load_choice'], 
                            filepaths['models_path'], cropped_paths['test'], 
                            train_test_options['batch_size'], train_test_options['image_size'], 
                            train_test_options['num_workers'], train_test_options['use_num_worker_mult'])
            elif configs['mode'] == "plot":
                train_test_options = configs['train_test_options']
                plot.plot_history(train_test_options['model_name'], filepaths['models_path'])
            else:
                print("Unexpected error in configs['mode'], this should have been caught by config_checker!")

if len(sys.argv) == 2:
    driver(sys.argv[1])
else:
    print("Proper usage: python main.py CONFIG_FILE.yml")
