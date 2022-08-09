import os
import PIL
from PIL import Image
from PIL import ImageChops
import pandas as pd
from pandas import DataFrame
import shutil
import re
import glob
import matplotlib.pyplot as plt
import numpy as np
import itertools

CUDA_VISIBLE_DEVICES = "-1"

# global variables
PATH = 'D:\Physics masters project\SCFI_data\SCFI_data\dataset2'  # where all data folders should be
BACTERIA_LIST = ['MS11', 'MX2', 'SN2']  # this should be at the start of any folders containing eg kanamycin data
BACTERIA_LIST = BACTERIA_LIST[:1]

def convert_to_npy(source, destination):
    """Given a source dir containing TIF files (at any level) copys dir structure to destination and populates
    with images converted to JPG format
    Plot compare is boolean for whether you want comparison of different dividers for saving jpeg
    """
    print('Converting to npys: ', source)
    count = 0
    for (root, dirs, files) in os.walk(source):
        count += 1
        print('Processing folder ' + str(count) + ' out of 106')
        print(root)
        for file in files:
            if str(file).endswith('.TIF'):
                im = Image.open(os.path.join(root, file))
                im = np.array(im)
                file_path = os.path.join(destination, os.path.relpath(root, source))
                os.makedirs(file_path, exist_ok=True)
                np.save(os.path.join(file_path, file[:-4] + '.npy'), im)


def copy_centres_file(source, destination):
    """Finds the text file containing the image centres and copies it to the new jpg folder"""
    print('Copying centre file')
    dirs = os.listdir(source)
    print(dirs)
    for dir_ in dirs:
        if str(dir_).endswith('.csv'):
            print(os.path.join(source, dir_))
            print(os.path.join(destination, dir_))
            shutil.copyfile(os.path.join(source, dir_), os.path.join(destination, dir_))


def convert_all_folders():
    """Searches PATH for folders prefixed with ANTIBIOTIC and applies the convert_to_jpg function"""
    for (root, dirs, files) in os.walk(PATH):
        # search dirs in PATH for those containing relevant data to our ANTIBIOTIC
        # I think this would work better using root and not dirs
        # print(root, dirs, files)
        for dir_ in dirs:
            dir_ = str(dir_)
            for i in range(len(BACTERIA_LIST)):
                # endswith condition prevents duplicating already converted folders
                if dir_.startswith(BACTERIA_LIST[i]) and not dir_.endswith(('jpg', 'crop','_npy')):
                    FULL_PATH = os.path.join(PATH, dir_)
                    convert_to_npy(FULL_PATH, FULL_PATH + '_npy')
                    copy_centres_file(FULL_PATH, FULL_PATH + '_npy')

def fix_dataset2_MS11():
    """
    Run this exactly once!!!!!!!!!!!!!
    :return:
    """
    fault_path_img = 'D:\Physics masters project\SCFI_data\SCFI_data\dataset2\MS11\images'
    fault_path_vid = 'D:\Physics masters project\SCFI_data\SCFI_data\dataset2\MS11\\videos'

    for file in os.listdir(fault_path_img):
        file_number = int(file[-8:-4])
        if file_number>48:
            if len(str(file_number)) == 3:
                file_number-=1
                new_file = file[:7] + str(file_number) + '_.TIF'
                print(file, new_file)
            else:
                file_number -= 1
                new_file = file[:8] + str(file_number) + '_.TIF'
                print(file, new_file)
            old_path = os.path.join(fault_path_img, file)
            new_path = os.path.join(fault_path_img, new_file)
            os.rename(old_path, new_path)
            print(new_path)

    for file in os.listdir(fault_path_img):
        if str(file).endswith('_.TIF'):
            os.rename(os.path.join(fault_path_img, file),os.path.join(fault_path_img, file[:10]+'.TIF'))

    # for dir in os.listdir(fault_path_vid):
    #     dir = str(dir)
    #     if dir.startswith('video'):
    #         dir_number = int(dir[5:])
    #         if dir_number > 48:
    #             dir_number -= 1
    #             new_folder = dir[:5] + str(dir_number)
    #             print(dir, new_folder)
    #             old_path = os.path.join(fault_path_vid, dir)
    #             new_path = os.path.join(fault_path_vid, new_folder+'_')
    #             os.rename(old_path, new_path)
    #             print(new_path)
    #
    # for dir in os.listdir(fault_path_vid):
    #     if str(dir).endswith('_'):
    #         os.rename(os.path.join(fault_path_vid, dir), os.path.join(fault_path_vid, dir[:-1]))


fix_dataset2_MS11()
convert_all_folders()

def centre(filepath, min_number):
    """Function which pulls the centre coordinates for the current video being cropped"""
    dirname = os.path.dirname(filepath)
    file_number = number_from_filename(dirname, min_number)
    [parent, subfolder] = os.path.split(os.path.split(dirname)[0])
    for files in os.listdir(parent):
        if str(files).endswith('centres.csv'):
            df = pd.read_csv(os.path.join(parent, files))
            # print(df)
    centre = df.values[file_number][:2]

    return centre


def number_from_filename(folder, min_number):
    name = os.path.basename(folder)
    split = re.split(r'[a-zA-Z]+', name)
    number = int(split[1])

    number -= min_number + 1
    return number


def centring_centres(root, file_name, bac_centre, im, cropped_size, bright_field_image):
    """Crops the image according to a centre supplied by the 'centre' function
    Plots and saves the uncropped image with the crops marked, then the bright field,
    then the cropped image.
    """
    # bright_field = np.load()
    shave_left = int(bac_centre[0] - cropped_size / 2)
    shave_right = int(bac_centre[0] + cropped_size / 2)
    shave_bottom = int(bac_centre[1] - cropped_size / 2)
    shave_top = int(bac_centre[1] + cropped_size / 2)
    if shave_left < 0:
        shave_left = 0
        shave_right = 200
    if shave_bottom < 0:
        shave_bottom = 0
        shave_top = 200
    if shave_right > 301:
        shave_left = 100
        shave_right = 301
    if shave_top > 301:
        shave_bottom = 100
        shave_top = 301

    crop = im[shave_bottom:shave_top, shave_left:shave_right]
    fig = plt.figure()
    plt.imshow(im)
    plt.title('Uncropped')
    y = np.linspace(0, 300, 100)
    x = np.array([shave_left] * 100)
    plt.plot(x, y, linestyle='-')
    y = np.linspace(0, 300, 100)
    x = np.array([shave_right] * 100)
    plt.plot(x, y, linestyle='-')
    x = np.linspace(0, 300, 100)
    y = np.array([shave_bottom] * 100)
    plt.plot(x, y, linestyle='-')
    x = np.linspace(0, 300, 100)
    y = np.array([shave_top] * 100)
    plt.plot(x, y, linestyle='-')
    plt.scatter([bac_centre[0]], [bac_centre[1]], marker='x')

    plt.ylim(300, 0)
    plt.xlim(0, 300)
    fig.savefig(os.path.join(root, 'figures', file_name + '_orig.png'))
    plt.close(fig)

    if file_name.endswith('0001'):
        fig2 = plt.figure()
        plt.imshow(bright_field_image)
        plt.title('Bright field')
        plt.scatter([bac_centre[0]], [bac_centre[1]], marker='x', c='black')
        fig2.savefig(os.path.join(root, 'figures', file_name + '_bf.png'))
        plt.close(fig2)

    fig3 = plt.figure()
    plt.imshow(crop, vmin=np.min(im), vmax=np.max(im))
    plt.title('Cropped (centre:' + str(bac_centre[0]) + ', ' + str(bac_centre[1]) + ')')
    plt.scatter([bac_centre[0] - shave_left], [bac_centre[1] - shave_bottom], marker='x', c='black')
    fig3.savefig(os.path.join(root, 'figures', file_name + '_crop.png'))
    plt.close(fig3)
    return crop


def cropping(source, destination):
    """Uses 'centring_centres' to crop each image in the source and moves it to new destination directory"""
    print('Beginning crop: ' + source)
    image_list = []
    for (root, dirs, files) in os.walk(source):
        print('Current folder:', root)
        if str(root).endswith('videos'):
            videos_in_folder = np.array(dirs)
            for i in range(len(videos_in_folder)):
                videos_in_folder[i] = int(videos_in_folder[i][5:])
            videos_in_folder = videos_in_folder.astype(int)
            min_folder_number = np.min(videos_in_folder)
            print('Minimum is:'+str(min_folder_number))
            count_files = 0
            dirs = np.sort(dirs)
            print(dirs)
        for file in files:
            if str(file).endswith('.npy') and str(file).startswith('video'):
                # creating paths
                dest_path = os.path.join(destination, os.path.relpath(root, source))
                file_path = os.path.join(root, file)
                os.makedirs(os.path.join(dest_path, 'figures'), exist_ok=True)
                # finding number of dataset from file
                if root[-3] == 'o':
                    dataset_number = str('00') + root[-2:]
                else:
                    dataset_number = str(0) + root[-3:]
                # loading bright_field
                bright_field = np.load(os.path.join(source, 'images', 'images' + dataset_number + '.npy'))
                # loading video frame
                im = np.load(file_path)
                # loading centre_coords
                im_centre = centre(file_path, min_folder_number)
                # cropping and saving figs
                cropped = centring_centres(dest_path, file[:-4], im_centre, im, 200,
                                           bright_field)

                np.save(os.path.join(dest_path, file), cropped)


for (root, dirs, files) in os.walk(PATH):
    # search dirs in PATH for those containing relevant data to our ANTIBIOTIC
    for dir_ in dirs:
        dir_ = str(dir_)
        for i in range(len(BACTERIA_LIST)):
            if dir_.startswith(BACTERIA_LIST[i]) and dir_.endswith('npy'):
                cropping(os.path.join(PATH, BACTERIA_LIST[i] + '_npy'),
                         os.path.join(PATH, BACTERIA_LIST[i] + '_npy', 'video_cropped'))
