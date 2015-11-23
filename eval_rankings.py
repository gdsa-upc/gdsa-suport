import os
import pandas as pd
import numpy as np
from utils.params import get_params
import matplotlib.pyplot as plt
import cv2

def display(params,query_id,ranking,relnotrel):
    
    ''' Display the first elements of the ranking '''
    
    # Read query image
    query_im =  cv2.imread(os.path.join(params['root'],params['database'],params['split'], 'images',query_id.split('.')[0] + '.jpg'))
    
    # Handling the duality in file terminations. I know it's not pretty, but it works...
    if query_im is None:
        query_im =  cv2.imread(os.path.join(params['root'],params['database'],params['split'], 'images',query_id.split('.')[0] + '.JPG'))
    
    # Blue contour for the query
    
    query_im = cv2.cvtColor(query_im,cv2.COLOR_BGR2RGB)
    query_im = cv2.copyMakeBorder(query_im,100,100,100,100,cv2.BORDER_CONSTANT,value=[0,0,255])
    # Init figure
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(4, 4, 1)
    
    # Display
    ax.imshow(query_im)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    
    # We will show the first 15 elements of the ranking
    for i in range(15):
                
        # Read image
        
        im =  cv2.imread(os.path.join(params['root'],params['database'],'train','images',ranking[0].tolist()[i] + '.jpg'))
        
        # Handling the duality in file terminations. I know it's not pretty, but it works...
        if im is None:
            
            im =  cv2.imread(os.path.join(params['root'],params['database'],'train', 'images',ranking[0].tolist()[i] + '.JPG'))
        
        # Switch to RGB to display with matplotlib
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        
        # Paint the boundaries with the ground truth 
        
        # If it was correctly selected
        if relnotrel[i] == 1:
            # Put green contour
            im = cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_CONSTANT,value= [0,255,0])

        # If it was not
        else:
            # Put red contour
            im = cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_CONSTANT,value= [255,0,0])
        
        # Show in figure
        ax = fig.add_subplot(4, 4, i+2)
        ax.imshow(im)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    
    print "Displaying..."
    plt.show()

def read_annotation(params):
    
    # Get true annotations
    annotation_val = pd.read_csv(os.path.join(params['root'],params['database'],params['split'],'annotation.txt'), sep='\t', header = 0)
    annotation_train = pd.read_csv(os.path.join(params['root'],params['database'],'train','annotation.txt'), sep='\t', header = 0)
    
    return annotation_val,annotation_train
    
def get_hitandmiss(ranking,query_class,annotation_train):
    
    # Initialize hit/miss list
    relnotrel = []
            
    # For each image id in the ranking...
    for i in ranking[0].tolist():
                
        # Get its class from the training annotations
        i_class = list(annotation_train.loc[annotation_train['ImageID'] == i]['ClassID'])[0]
                                
        # And if it matches the query class...
        if query_class == i_class:
                    
            # Then it means it's correct
            relnotrel.append(1)
        else:
                    
            # If it doesn't, then we failed :(
            relnotrel.append(0)
        
    return relnotrel

def AveragePrecision(relist):
    '''Takes a hit & miss list with ones and zeros and computes its average precision'''

    # Initialize the accumulated sum of precisions
    accu = 0
    
    # Initialize the number of correct instances found so far
    numRel = 0
    
    # For all elements in the hit & miss list
    for k in range(len(relist)):
        
        # If the value is 1
        if relist[k] == 1:
            
            # We add 1 to the number of correct instances
            numRel = numRel + 1
            
            # We calculate the precision at k (+1 because we start at 0) and we accumulate it
            accu += float( numRel )/ float(k+1)

    # When we finish, we divide by the total number of relevant instances, which is the sum of ones in the list
    return (accu/np.sum(relist))

def load_ranking(params,query_id, annotation_val):
    
    ''' Loads and  returns the ranking from the txt. Returns the true class of the query image as well.'''
    
    # Get the true class of the validation image for which we will evaluate the ranking
    query_class = list(annotation_val.loc[annotation_val['ImageID'] == query_id.split('.')[0]]['ClassID'])[0]
        
    # Open its ranking file
    ranking = pd.read_csv(os.path.join(params['root'],params['root_save'],params['rankings_dir'],params['descriptor_type'],params['split'],query_id.split('.')[0] + '.txt'),header= None)
    
    return query_class, ranking
    

def eval_rankings(params):
    
    ap_list = []
    
    # Get true annotations
    annotation_val, annotation_train = read_annotation(params)
    
    # For all generated rankings
    for val_id in os.listdir(os.path.join(params['root'],params['root_save'],params['rankings_dir'],params['descriptor_type'],params['split'])):
        
        query_class, ranking = load_ranking(params,val_id,annotation_val)
        
        # We do not evaluate the queries in the unknown class ! 
        if not query_class == "desconegut":
            
            
            # Get the hit & miss list
            relnotrel = get_hitandmiss(ranking,query_class,annotation_train)
            
            # Calculate average precision of the list
            ap = AveragePrecision(relnotrel)
                        
            # Store it
            ap_list.append(ap)
            

    return ap_list
    
   
if __name__ == "__main__":
    
    params = get_params()
    
    ap_list = eval_ranking(params)
    
    print np.mean(ap_list)