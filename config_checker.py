valid_modes = ['crop', 'verify', 'train', 'test', 'plot']
valid_data_types = ['all', 'train', 'val', 'test']
valid_remove_extra = [True, False]
valid_reset_progress = [True, False]
valid_crop_missing = [True, False]
valid_models = ["alexnet",
                "vgg11",
                "vgg11_bn",
                "vgg13",
                "vgg13_bn",
                "vgg16",
                "vgg16_bn",
                "vgg19",
                "vgg19_bn",
                "resnet18",
                "resnet34",
                "resnet50",
                "resnet101",
                "resnet152",
                "squeezenet1_0",
                "squeezenet1_1",
                "densenet121",
                "densenet161",
                "densenet169",
                "densenet201",
                "inception_v3",
                "googlenet",
                "shufflenet_v2_x0_5",
                "shufflenet_v2_x1_0",
                "shufflenet_v2_x1_5",
                "shufflenet_v2_x2_0",
                "resnext50_32x4d",
                "resnext101_32x8d",
                "wide_resnet50_2",
                "wide_resnet101_2",
                "mobilenet_v2",
                "ResNeXt-101-32x8d"]
valid_pretrained = [True, False]
valid_use_num_worker_mult = [True, False]

def is_empty(name, target, config_mode):
    """Checks if the target is not empty."""
    if target == None:
        print(f"{name} cannot be empty for {config_mode} mode!")
        return True
    return False

def is_instance_string(name, target, config_mode):
    """Checks if the target is or can be a string."""
    if is_empty(name, target, config_mode):
        return False
    else:
        if not isinstance(target, (str, int)):
            print(f"{name} must be a string!")
            return False
    return True

def is_endswith_ending(name, target, ending, config_mode):
    """Checks if the target ends with ending"""
    if is_empty(name, target, config_mode):
        return False
    else:
        if not target.endswith(ending):
            print(f"{target} must end with a {ending}\n")
            return False
    return True

def is_in(name, target, array, config_mode):
    """Checks if the target is in the array."""
    if is_empty(name, target, config_mode):
        return False
    else:
        if not target in array:
            print(f"{name} must be in {array}!")
            return False
    return True

def is_instance_int(name, target, config_mode):
    """Checks if the target is an integer."""
    if is_empty(name, target, config_mode):
        return False
    else:
        if not isinstance(target, int):
            print(f"{name} must be an integer!")
            return False
    return True

def check_filepaths_child(filepaths, pathtype, datatype, is_folder, config_mode):
    """Checks if the children of filepaths can be accessed.
        Assumes that if a filepath is not a folder, it must be a .json"""
    all_passed = True
    try:
        paths = filepaths[pathtype]
    except:
        print(f"Invalid {pathtype} structure!")
        paths = []
        all_passed = False
    if datatype != 'all' and datatype in valid_data_types:
        name = f"{pathtype}:{datatype}"
        target = filepaths[pathtype][datatype]

        all_passed = is_instance_string(name, target, config_mode) and all_passed
        if is_folder:
            ending = '/'
        else:
            ending = '.json'
        all_passed = is_endswith_ending(name, target, ending, config_mode) and all_passed

    else:
        for each in valid_data_types[1:]:
            name = f"{pathtype}:{each}"
            target = filepaths[pathtype][each]

            all_passed = is_instance_string(name, target, config_mode) and all_passed
            if is_folder:
                ending = '/'
            else:
                ending = '.json'
            all_passed = is_endswith_ending(name, target, ending, config_mode) and all_passed

    return all_passed

def check_crop_paths(filepaths, datatype, config_mode):
    """Checks if the entries in filepaths that are needed for crop mode are valid."""
    all_passed = True

    all_passed = check_filepaths_child(filepaths, 'data_paths', datatype, True, config_mode) and all_passed

    all_passed = check_filepaths_child(filepaths, 'cropped_paths', datatype, True, config_mode) and all_passed

    all_passed = check_filepaths_child(filepaths, 'instances_paths', datatype, False, config_mode) and all_passed

    return all_passed

def check_verify_paths(filepaths, datatype, config_mode):
    """Checks if the entries in filepaths that are needed for verify mode are valid."""
    all_passed = True

    all_passed = check_crop_paths(filepaths, datatype, config_mode) and all_passed

    all_passed = check_filepaths_child(filepaths, 'verify_progress_paths', datatype, False, config_mode) and all_passed

    return all_passed

def check(configs):
    """Checks if the entries based on the mode are valid."""
    all_passed = True

    try:
        filepaths = configs['filepaths']
    except:
        print("Invalid filepaths structure!")
        all_passed = False
    
    all_passed = is_in("mode", configs['mode'], valid_modes, 'mode') and all_passed

    if configs['mode'] == 'crop':
        config_mode = 'crop'
        # Checks if the entries in filepaths that are needed for crop mode are valid.
        try:
            crop_options = configs['crop_options']
        except:
            print("Invalid crop_options structure!")
            all_passed = False

        all_passed = is_in("dataset_choice", crop_options['dataset_choice'], valid_data_types, config_mode) and all_passed

        all_passed = check_crop_paths(filepaths, crop_options['dataset_choice'], config_mode) and all_passed

    elif configs['mode'] == 'verify':
        config_mode = 'verify'
        # Checks if the entries in filepaths that are needed for verify mode are valid.
        try:
            verify_options = configs['verify_options']
        except:
            print("Invalid verify_options structure!")
            all_passed = False
        all_passed = is_in("dataset_choice", verify_options['dataset_choice'], valid_data_types, config_mode) and all_passed
        all_passed = is_in("remove_extra", verify_options['remove_extra'], valid_remove_extra, config_mode) and all_passed
        all_passed = is_in("reset_progress", verify_options['reset_progress'], valid_reset_progress, config_mode) and all_passed
        all_passed = is_instance_int('verify_save_step', verify_options['verify_save_step'], config_mode) and all_passed
        all_passed = is_in("crop_missing", verify_options['crop_missing'], valid_crop_missing, config_mode) and all_passed

        all_passed = check_verify_paths(filepaths, verify_options['dataset_choice'], config_mode) and all_passed

    elif configs['mode'] == 'train' or configs['mode'] == 'test' or configs['mode'] == 'plot':
        all_passed = is_instance_string("models_path", filepaths['models_path'], 'train, test or plot') and all_passed
        all_passed = is_endswith_ending('models_path', filepaths['models_path'], '/', 'train, test or plot') and all_passed
        
        # Checks if the entries in filepaths that are needed for train or test mode are valid.
        try:
            train_test_options = configs['train_test_options']
        except:
            print("Invalid train_test_options structure!")
            all_passed = False

        all_passed = is_in("model_name", train_test_options['model_name'], valid_models, 'train, test or plot') and all_passed

        if configs['mode'] == 'train':
            config_mode = 'train'

            check_filepaths_child(filepaths, 'cropped_paths', 'train', True, config_mode)
            check_filepaths_child(filepaths, 'cropped_paths', 'val', True, config_mode)
        elif configs['mode'] == 'test':
            config_mode = 'test'
            
            check_filepaths_child(filepaths, 'cropped_paths', 'test', True, config_mode)

        if configs['mode'] != 'plot':
            config_mode = 'train or test'

            all_passed = is_instance_int('load_choice', train_test_options['load_choice'], config_mode) and all_passed
            all_passed = is_instance_int('batch_size', train_test_options['batch_size'], config_mode) and all_passed
            all_passed = is_instance_int('image_size', train_test_options['image_size'], config_mode) and all_passed
            all_passed = is_instance_int('num_workers', train_test_options['num_workers'], config_mode) and all_passed
            all_passed = is_in("use_num_worker_mult", train_test_options['use_num_worker_mult'], valid_use_num_worker_mult, config_mode) and all_passed

            if configs['mode'] == 'train':
                config_mode = 'train'
                all_passed = is_in("pretrained", train_test_options['pretrained'], valid_pretrained, config_mode) and all_passed
                all_passed = is_instance_int('num_epochs', train_test_options['num_epochs'], config_mode) and all_passed
    else:
        print("Error in config_checker.check() mode checking.")

    if not all_passed:
        print("One or more errors in config file!")

    return all_passed