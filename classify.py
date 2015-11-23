import os
import random
import pickle
from utils.params import get_params

def classify(params):
    
    val_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],params['split'] + params['descriptor_type'] + str(params['descriptor_size']) + '.p'),'rb'))
    
    outfile = open(os.path.join(params['root'],params['root_save'],params['classification_dir'],params['descriptor_type'], params['split'] + '_classification.txt'),'w')
    
    for val_id in val_features.keys():
    
        # Pick a random number from 0 to 12
        pos = random.randint(0,len(params['possible_labels'])-1)
        
        # Take the class label at this position
        label = params['possible_labels'][pos]
        
        # Write the prediction to file
        outfile.write(val_id.split('.')[0] + '\t' + label + '\n')
    
    outfile.close()
    
    
    
if __name__ == "__main__":
    
    params = get_params()
    
    classify(params)