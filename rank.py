import os
import random
import pickle
from utils.params import get_params

def rank(params):
    
    # Load train and validation feature dictionaries
    val_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],params['split'] + params['descriptor_type'] + str(params['descriptor_size']) + '.p'),'rb'))
    train_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],'train' + params['descriptor_type'] + str(params['descriptor_size']) + '.p'),'rb'))
    
    # We ignore the features at this point, we just want the image ids for the training set and sort them randomly
    
    # For each image id in the validation set
    for val_id in val_features.keys():
        
        print val_id.split('.')[0]
       
        # The ranking is composed with the ids of all training images
        ranking = train_features.keys()
        
        
        # Sorting them randomly
        random.shuffle(ranking)
        
        # Instead of a shuffle, we could have used distances between the query descriptor and
        # all training descriptors for sorting. Since descriptors are random, results should be equally bad.
        
        outfile = open(os.path.join(params['root'],params['root_save'],params['rankings_dir'],params['descriptor_type'],params['split'],val_id.split('.')[0] + '.txt'),'w')
        
        for item in ranking:
            
            outfile.write(item.split('.')[0] + '\n')
        
        outfile.close()

if __name__ == "__main__":
    
    params = get_params()
    rank(params)