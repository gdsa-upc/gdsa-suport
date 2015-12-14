import os
import random
import numpy as np
import pandas as pd
import time
import pickle
from scipy.stats import itemfreq
from params import get_params
from sklearn import svm, grid_search
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

def classify(params):
    
    # Load validation features
    val_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],
                             params['split'] + "_" + str(params['descriptor_size']) + "_"
                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p'),'rb'))
                             
    # Open output file                        
    outfile = open(os.path.join(params['root'],params['root_save'],params['classification_dir'],params['descriptor_type'], params['split'] + '_classification.txt'),'w')
    
    # Classifier
    clf = pickle.load(open(os.path.join(params['root'],params['root_save'],params['classifiers_dir'],
                                        params['classifier'] + "_" + str(params['descriptor_size']) + "_"
                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p'),'rb'))
    
    # Store predictions for all validation images
    for index in range(len(val_features.keys())):
        
        # Get image name
        val_id = val_features.keys()[index]
        
        # Classifier prediction
        prediction = clf.predict(val_features[val_id])
        
        # Write line to file
        outfile.write(val_id.split('.')[0] + "\t" + prediction[0] + "\n")
        
    outfile.close()

def get_training_data(params):
    
    # Load training features
    train_features = pickle.load(open(os.path.join(params['root'],params['root_save'],params['feats_dir'],
                             'train' + "_" + str(params['descriptor_size']) + "_"
                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p'),'rb'))
                             
                             
    # Load training annotations
    annotation_train = pd.read_csv(os.path.join(params['root'],params['database'],'train','annotation.txt'), sep='\t', header = 0)
    
    # Create list of labels and array of features
    
    labels = []
    features = []
    
    # Randomly sort names
    names = train_features.keys()
    
    random.shuffle(names)
    
    # Keep track of the desconegut class
    desconegut_count = 0
    
    # Find the label for each image id
    for i in names:

        # Get its class
        i_class = list(annotation_train.loc[annotation_train['ImageID'] == i.split('.')[0]]['ClassID'])[0]
    
        # Manually balance classes: Get as many samples from the desconegut class as the other classes (25)
        if params['manual_balance']:
        
            # Everytime we find a desconegut instance, we increase the counter
            if i_class == 'desconegut':
                desconegut_count += 1
                
            # Then, check how many we stored already    
            if i_class == 'desconegut' and desconegut_count > 25:
                
                # If we stored 25 desconegut instances, and a new one comes, we pass
                pass
            else:
                
                # Else we save it in the lists normally
                labels.append(i_class)
                features.append(train_features[i]) 
                 
        else:
            
            # No manual class balancing. All features are used.
            labels.append(i_class)
            features.append(train_features[i]) 
            
    features = np.array(features).squeeze()

    if not params['manual_balance']:
        # Include class weights if we don't balance classes manually.
        class_weights = get_class_weights(labels)
    else:
        class_weights = None

    return  features , labels, class_weights

def get_classifier(params,class_weights = None):
    

    if params['classifier'] == 'SVM':

        clf = svm.SVC(class_weight=class_weights)

        
    elif params['classifier'] == 'KNN':
        
        clf = KNeighborsClassifier(params['num_neighbors'])
        
    return clf
        

def get_class_weights(labels):

    # Get number of times each class appears in the training data
    freq = itemfreq(labels)

    # Isolate values and convert to integer
    freq = np.array(freq[:,1]).astype(int)

    # Get individual class weights. If a class has more samples, its weight is smaller
    freq = float(len(labels))/ (13 * freq)

    # Get unique class names
    class_names = np.unique(labels)

    # Init dictionary
    class_weights = {}

    # Put class weights in dictionary
    for i in range(len(class_names)):

        class_weights[class_names[i]] = freq[i]

    return class_weights


def tune_parameters(params,X,y,clf):
    
    # Initialize the gridsearch object
    gs = grid_search.GridSearchCV(clf, params['svm_tune'],cv=5,scoring='f1_macro')
    
    # Fit data
    gs.fit(X,y)
    
    print "Chosen parameters", gs.best_params_
    print "Score during tuning:", gs.best_score_
    
    return gs.best_estimator_

            
def train_classifier(params):
    
    # Load training data
    X,y, class_weights = get_training_data(params)
    
    # Initialize the classifier
    clf = get_classifier(params,class_weights)
    
    # Tune parameters
    
    if params['classifier'] == 'SVM':
        clf = tune_parameters(params,X,y,clf)

    
    # Fit data to our model
    clf.fit(X,y)
    
    # Save model to disk
    pickle.dump(clf,open(os.path.join(params['root'],params['root_save'],params['classifiers_dir'],
                                        params['classifier'] + "_" + str(params['descriptor_size']) + "_"
                             + params['descriptor_type'] + "_" + params['keypoint_type'] + '.p'),'wb'))
    
if __name__ == "__main__":
    
    params = get_params()
    
    print "Training classifier..."
    t = time.time()
    train_classifier(params)
    print "Done. Time elapsed:", time.time() - t
    
    classify(params)
