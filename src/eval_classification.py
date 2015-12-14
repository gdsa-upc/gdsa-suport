import os, math, sys
import pandas as pd
import numpy as np
import cv2
from params import get_params

# We need to add the source code path to the python path if we want to call modules such as 'utils'
params = get_params()
sys.path.insert(0,params['src'])

import utils.kaggle_scripts as kaggle_scripts
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score,precision_score,recall_score,accuracy_score
import warnings
warnings.filterwarnings("ignore")


# Adapted function from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py

def plot_confusion_matrix(cm, true_labels,normalize = False,title='Confusion matrix', cmap=plt.cm.Blues):
    
    # Normalize matrix rows to sum 1
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(true_labels))
    plt.xticks(tick_marks, true_labels, rotation=90)
    plt.yticks(tick_marks, true_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.show()

def load_annotation(params):

    # Load the true annotations
    annotation_val = pd.read_csv(os.path.join(params['root'],params['database'],params['split'],'annotation.txt'), sep='\t', header = 0)
    true_labels = annotation_val['ClassID'].tolist()
    names = annotation_val['ImageID'].tolist()

    # Sort in ascending order for correspondance with predictions
    true_labels = list(np.array(true_labels)[np.argsort(np.array(names))])

    names = list(np.array(names)[np.argsort(np.array(names))])

    return true_labels, names

def load_predictions(params):
    # Load the predictions
    prediction_file = pd.read_csv(os.path.join(params['root'],params['root_save'],params['classification_dir'],params['descriptor_type'], params['split'] + '_classification.txt'), sep='\t', header = None)
    prediction_labels = prediction_file[1].tolist()
    prediction_names = prediction_file[0].tolist()

    # Sort in ascending order for correspondence with Ground Truth Annotations
    prediction_labels = list(np.array(prediction_labels)[np.argsort(np.array(prediction_names))])
    prediction_names =  list(np.array(prediction_names)[np.argsort(np.array(prediction_names))])
    
    return prediction_labels, prediction_names

def eval_classification(params):
    
    # Load the true annotations
    true_labels, names = load_annotation(params)

    # Load predictions
    prediction_labels, prediction_names = load_predictions(params)

    # Save predictions in Kaggle friendly format
    if params['save_for_kaggle']:
        file_to_save = open(os.path.join(params['root'],params['root_save'],params['kaggle_dir'],params['descriptor_type'] + '_' + params['split'] + '_classification.csv'),'w')
        kaggle_scripts.save_classification_file(file_to_save, prediction_names, prediction_labels)

    # Compute evaluation metrics
    cm = confusion_matrix(true_labels,prediction_labels)
    
    # By settin average = None we are computing the metrics for each class independently.
    
    # We could choose to average them with equal weights, or with different weights
    # according to the number of instances of each class. Check the documentation for details.
    f1 = f1_score(true_labels,prediction_labels,average=None)
    precision = precision_score(true_labels,prediction_labels,average=None)
    recall = recall_score(true_labels,prediction_labels,average=None)
    accuracy = accuracy_score(true_labels,prediction_labels)
    
    
    return f1, precision, recall, accuracy,cm, np.unique(true_labels)
    
def plot_class(params, class_name):

    true_labels, names = load_annotation(params)
    prediction_labels, prediction_names = load_predictions(params)

    imnames = np.array(prediction_names)[np.array(prediction_labels) == class_name]

    print "Number of images:",len(imnames)
    # Init figure
    fig = plt.figure(figsize=(20,10))

    for i in range(len(imnames)):

        # Read image
        im =  cv2.imread(os.path.join(params['root'],params['database'],params['split'],'images',imnames[i] + '.jpg'))

        # Handling the duality in file terminations. I know it's not pretty, but it works...
        if im is None:

            im =  cv2.imread(os.path.join(params['root'],params['database'],params['split'], 'images', imnames[i] + '.JPG'))

        # Switch to RGB to display with matplotlib
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

        # Include GT class

        # Check the true class of the current image
        true_class = np.array(true_labels)[np.array(names) == imnames[i]]

        # Paint image borders according to the true class
        if true_class == class_name:
            # Put green contour
            im = cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_CONSTANT,value= [0,255,0])
        else:
            # Put red contour
            im = cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_CONSTANT,value= [255,0,0])

        ax = fig.add_subplot(math.ceil(float(len(imnames))/4),4, i+1)
        ax.imshow(im)
        ax.set_title(true_class)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    plt.show()

if __name__ == "__main__":
    
    params = get_params()
    
    f1, precision, recall, accuracy,cm, labels = eval_classification(params)
    
    print "F1:", np.mean(f1)
    print "Precision:", np.mean(precision)
    print "Recall:", np.mean(recall)
    print "Accuracy:", accuracy
    
    plot_confusion_matrix(cm, labels,normalize = True)

    '''
    class_name = 'farmacia_albinyana'
    print "Displaying images labeled as", class_name,'...'
    t = time.time()
    plot_class(params,class_name)
    print "Done. Time elapsed:", time.time() - t
    '''