#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[103]:


PATH = '/D:/'  # where all data folders should be
BACTERIA_LIST = ['kanamycin']  # this should be at the start of any folders containing eg kanamycin data
BACTERIA_LIST = BACTERIA_LIST[:1]

def convert_to_npy(source, destination):
    """Given a source dir containing TIF files (at any level) copys dir structure to destination and populates
    with images converted to JPG format
    Plot compare is boolean for whether you want comparison of different dividers for saving jpeg
    """
    print('Converting to npys: ', source)
    for (root, dirs, files) in os.walk(source):
        print('Processing folder ' +str(root))
        root_name = str(os.path.split(root)[1])
        if str(root_name).startswith('video') and not(str(root_name).endswith('s')):
            new_old_folder = str(os.path.split(os.path.dirname(os.path.dirname(root)))[1])
            treated_resistance_folder = str(os.path.split(os.path.dirname(os.path.dirname(os.path.dirname(root))))[1])
            if str(new_old_folder).startswith('old'):
                sub_folder = 'dataset_1'
            else:
                sub_folder = 'dataset_2'
            
            print(treated_resistance_folder)
            if  str(treated_resistance_folder).startswith('treat resistance'):
                folder = 't_r'
            if  str(treated_resistance_folder).startswith('untreat resistance'):
                folder = 'u_r'
            if  str(treated_resistance_folder).startswith('treat susceptible'):
                folder = 't_s'
            if  str(treated_resistance_folder).startswith('untreat susceptible'):
                folder = 'u_s'
            for file in files:
                if str(file).endswith('.TIF'):
                    im = Image.open(os.path.join(root, file))
                    im = np.array(im)
                    file_path = os.path.join(destination, folder, sub_folder, 'videos', root_name)
                    os.makedirs(file_path, exist_ok=True)
                    np.save(os.path.join(file_path,file[:-4] + '.npy'), im)


# In[105]:


test_path_list = ['D:\\kanamycin']
for i in range(len(test_path_list)):
    convert_to_npy(test_path_list[i],test_path_list[i]+'_npy')


# In[55]:


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


# In[56]:


convert_all_folders()


# In[ ]:




