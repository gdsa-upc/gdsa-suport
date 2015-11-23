import os
import pandas as pd
import numpy as np
from utils.params import get_params
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


def eval_classification(params):
    
    # Load the true annotations
    annotation_val = pd.read_csv(os.path.join(params['root'],params['database'],params['split'],'annotation.txt'), sep='\t', header = 0)
    true_labels = annotation_val['ClassID'].tolist()
    
    # Load the predictions
    prediction_file = pd.read_csv(os.path.join(params['root'],params['root_save'],params['classification_dir'],params['descriptor_type'], params['split'] + '_classification.txt'), sep='\t', header = None)
    prediction_labels = prediction_file[1].tolist()
    
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
    
    
    
if __name__ == "__main__":
    
    params = get_params()
    
    f1, precision, recall, accuracy,cm, labels = eval_classification(params)
    
    print np.mean(f1)