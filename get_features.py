from utils.params import get_params
import os
import random
import numpy as np
import pickle

def get_features(params):
    # Read image names
    with open(os.path.join(params['root'],params['root_save'],params['image_lists'],params['split'] + '.txt'),'r') as f:
        image_list = f.readlines()

    # Initialize feature dictionary
    features = {}

    # Calculate random descriptor for each image
    for image_name in image_list:

        features[image_name] = np.random.rand(params['descriptor_size'])

    # Save dictionary to disk with unique name
    save_file = os.path.join(params['root'],params['root_save'],params['feats_dir'],params['split'] + params['descriptor_type'] + str(params['descriptor_size']) + '.p')
    pickle.dump(features,open(save_file,'wb'))


if __name__ == "__main__":

    params = get_params()

    # Save features for validation set
    get_features(params)
    
    # Change to training set
    params['split'] = 'train'
    
    # Save features for training set
    get_features(params)
