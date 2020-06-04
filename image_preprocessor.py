import os
import json
import cv2
from PIL import Image

def crop_images(annotations, image_id_filename, categories, src_path, dst_path):
    """Crops all images in src_path into dst_path based on the bounding boxes in annotations.

    Args:
        annotations         (list): List of annotations containing various information of each image.
        image_id_filename   (dict): Dict of {image ids:image name}.
        categories          (dict): Dict of {category ids:category names}.
        src_path            (str): The folder path that contains the uncropped images.
        dst_path            (str): The folder path that will contain the cropped images.
    """
    total_files = len(annotations)
    counter = 0
    for each in annotations:
        image_id = each['image_id']
        image_name = image_id_filename[image_id]
        outpath = dst_path + categories[each['category_id']] + "/" + str(each['id']) + ".jpg"
        if not os.path.isfile(outpath):
            (x, y, w, h) = each['bbox']
            img = cv2.imread(src_path + image_name)
            if not img is None:
                cropped_img = img[int(y):int(y+h), int(x):int(x+w)]
                cv2.imwrite(outpath, cropped_img)
            else:
                print(f'Error reading {src_path + image_name}!')
        if counter % 5000 == 0:
            print(f'Progress: {100*counter/total_files}%')
        counter += 1
    print(f'Progress: {100*counter/total_files}%')

def crop_selection(instances, src_path, dst_path, selection=None):
    """Preprocesses instances file for cropping information, and selects files that need to be cropped.

    Args:
        instances       (dict): Dict of instances containing various information of each image.
        src_path        (str): The folder path that contains the uncropped images.
        dst_path        (str): The folder path that will contain the cropped images.
        selection       (list): List of images that will be cropped. All images are selected if selection=None.
    """
    images = instances['images']
    annotations = instances['annotations']
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    image_id_filename = {}
    for each in images:
        image_id_filename[each['id']] = each['file_name']
    categories = {}
    for each in instances['categories']:
        categories[each['id']] = each['name']
    for each in categories.values():
        if not os.path.isdir(dst_path + each):
            os.mkdir(dst_path + each)
    selected_annotations = []
    if selection is not None:
        for each in annotations:
            filename = str(each['id']) + ".jpg"
            if filename in selection:
                selected_annotations.append(each)
    else:
        selected_annotations = annotations
    crop_images(selected_annotations, image_id_filename, categories, src_path, dst_path)

def crop_images_driver(instance_path, uncropped_path, cropped_path):
    """Opens the instance file and starts the cropping process.

    Args:
        instance_path   (str): The filepath of a .json instance file.
        uncropped_path  (str): The folder path that contains the uncropped images.
        cropped_path    (str): The folder path that will contain the cropped images.
    """
    if os.path.isfile(instance_path):
        with open(instance_path) as jsonfile:
            instances = (json.load(jsonfile))
            print(instances.keys())
            crop_selection(instances, uncropped_path, cropped_path)
    else:
        print(f"{instance_path} not found! Cropping cannot be done.")

def remove_extra_expected(diff_files):
    """Deletes all files in listed in diff_files."""
    for each in diff_files:
        os.remove(each)

def file_num_compare(instance_path, uncropped_path, cropped_path, datatype, remove_extra=True, crop_missing=True):
    """Checks if the number of cropped images is equal to the number of entries in the annotations section of the instance file.
        If the numbers are not equal, the function finds and lists the missing or extra files.

    Attributes:
        expected_num    (int): The number of expected images which is also the number of entries in the annotations section of the instances file.
        found_num       (int): The number of cropped images found.

    Args:
        instance_path   (str): The filepath of a .json instance file.
        cropped_path    (str): The folder path that contains the cropped images to be checked.
        datatype        (str): The type of data being checked. E.g. Validation data
        remove_extra    (bool): If true, extra files that are unexpected will be automatically removed.
        crop_missing    (bool): If true, missing files that are unexpected will be automatically cropped.
    """
    if os.path.isdir(cropped_path):
        if os.path.isfile(instance_path):
            with open(instance_path) as jsonfile:
                instances = (json.load(jsonfile))
                expected_num = len(instances['annotations'])
                print(f"{datatype} Cropped Expected: {expected_num}")
                found_num = sum([len(files) for path, subdirs, files in os.walk(cropped_path)])
                print(f'{datatype} Cropped Found: {found_num}')
                if (expected_num != found_num):
                    expected_files = []
                    for each in instances['annotations']:
                        expected_files.append(str(each['id']) + ".jpg")
                    found_files = []
                    found_filepaths = {}
                    for path, subdirs, files in os.walk(cropped_path):
                        for name in files:
                            found_files.append(name)
                            found_filepaths[name] = os.path.join(path, name)
                    if len(expected_files) > len(found_files):
                        print(f"Missing expected files! Finding missing or mismatched files...")
                        diff_files = []
                        for each in expected_files:
                            if each not in found_files:
                                diff_files.append(each)
                        print(f"Missing expected or mismatched files: {diff_files}")
                        if crop_missing:
                            crop_selection(instances, uncropped_path, cropped_path, diff_files)
                    else:
                        print(f"Extra expected files! Finding extra or mismatched files...")
                        diff_files = []
                        for each in found_files:
                            if each not in expected_files:
                                diff_files.append(found_filepaths[each])
                        print(f"Extra expected or mismatched files: {diff_files}")
                        if remove_extra:
                            for each in diff_files:
                                os.remove(each)
        else:
            print(f"{instance_path} not found!")
    else:
        print(f"{cropped_path} not found!")

def verify_jpg(folderpath, progress_path, reset_progress, verify_save_step):
    """Verifies each image in folderpath, and saves verification progress in progress_path. 
        If progress_path exists when this function is called, verification will resume by ignoring file names lsited in progress_path.

    Args:
        folderpath          (str): The folder path that contains the cropped images to be checked.
        progress_path       (str): The filepath that contains the file names of all already checked images.
        reset_progress      (bool): If true, the progress_path file will be deleted and 
                                        the function will verify all images as there is no longer any saved progress file.
        verify_save_step    (int): The number of images to be checked each loop before saving.
    
    Returns:
        bad_files           (list): A list of images that either could not be opened or failed verification.
    """
    progress_list = []
    if reset_progress:
        if os.path.exists(progress_path):
            os.remove(progress_path)
    else:
        if os.path.exists(progress_path):
            try:
                with open(progress_path) as jsonfile:
                    progress_list = json.load(jsonfile)
            except (IOError, SyntaxError, json.JSONDecodeError):
                os.remove(progress_path)
    counter = 0
    bad_files = {}
    imagepaths = {}
    for path, subdirs, files in os.walk(folderpath):
        for name in files:
            imagepaths[name] = os.path.join(path, name)
    total_files = len(imagepaths)
    old_len = len(progress_list)
    if total_files > 0:
        print(f'Current Progress: {old_len / total_files * 100}%')
        for filename in imagepaths:
            if filename.endswith('.jpg'):
                if imagepaths[filename] not in progress_list:
                    try:
                        img = Image.open(imagepaths[filename]) # open the image file
                        img.verify() # verify that it is, in fact an image
                        progress_list.append(imagepaths[filename])
                        # print(f'Verified: {imagepaths[filename]}')
                    except Exception:
                        print(f'Bad file: {imagepaths[filename]}') # print out the names of corrupt files
                        bad_files[filename] = imagepaths[filename]
                counter += 1
                if counter > verify_save_step and old_len < len(progress_list):
                    with open(progress_path, 'w') as outfile:
                        json.dump(progress_list, outfile)
                    counter = 0
                    print(f'Progress: {len(progress_list) / total_files * 100}%')
                    old_len = len(progress_list)

        with open(progress_path, 'w') as outfile:
            json.dump(progress_list, outfile)
        counter = 0
        print(f'Progress: {len(progress_list) / total_files * 100}%')
        print(f'Number of bad files: {len(bad_files)}')
    else:
        print(f"No files found in {folderpath}.")
    return bad_files

def verify_jpg_driver(instance_path, uncropped_path, cropped_path, progress_path, reset_progress, verify_save_step):
    """Verifies the cropped images and recrops the bad files.

    Attributes:
        bad_files           (list): A list of images that either could not be opened or failed verification.

    Args:
        cropped_path        (str): The folder path that contains the cropped images to be checked.
        uncropped_path      (str): The folder path that contains the cropped images to be checked.
        cropped_path        (str): The folder path that contains the cropped images to be checked.
        progress_path       (str): The filepath that contains the file names of all already checked images.
        reset_progress      (bool): If true, the progress_path file will be deleted and 
                                        the function will verify all images as there is no longer any saved progress file.
        verify_save_step    (int): The number of images to be checked each loop before saving.
    """
    if os.path.exists(cropped_path):
        bad_files = [0] 
        while len(bad_files) > 0:
            bad_files = verify_jpg(cropped_path, progress_path, reset_progress, verify_save_step)
            if len(bad_files) > 0:
                for each in bad_files:
                    os.remove(bad_files[each])
                if os.path.isfile(instance_path):
                    with open(instance_path) as jsonfile:
                        instances = (json.load(jsonfile))
                        print(instances.keys())
                        crop_selection(instances, uncropped_path, cropped_path, bad_files)
                    print("reset_progressing after removing bad files...")
                else:
                    print(f"{instance_path} not found! Cropping cannot be done.")
                