import numpy as np
import pandas
import sys,os
import random
# Functions to save solution files in the correct format for Kaggle Competition

def save_ranking_file(file_to_save,image_id,ranking):

    '''
    :param file_to_save: name of the file to be saved
    :param image_id: name of the query image
    :param ranking: ranking for the image image_id
    :return: the updated state of the file to be saved
    '''

    # Write query name
    file_to_save.write(image_id.split('.')[0] + ',')

    # Convert elements to string and ranking to list
    ranking = np.array(ranking).astype('str').tolist()

    # Write space separated ranking
    for item in ranking:
        file_to_save.write(item[0] + " ")

    file_to_save.write('\n')

    return file_to_save

def write_kaggle(ann,image_id,file_to_save,ann_train,usage):

    i_class = list(ann.loc[ann['ImageID'] == image_id]['ClassID'])[0]
    if not i_class == 'desconegut':
        file_to_save.write(image_id + ',')

        if usage == '':
            to_write = ann_train['ImageID']
            random.shuffle(to_write)
        else:
            to_write = ann_train.loc[ann_train['ClassID'].isin([i_class])]['ImageID'].tolist()

        for i in to_write:
            file_to_save.write(i + ' ')

        if not usage=='':
            file_to_save.write(','+usage)
        file_to_save.write('\n')

def convert_ranking_annotation(ann_test, ann_val, ann_train,gt_file,random_file):

    # Convert ranking annotation and store it for Kaggle (only needed for teachers)
    gt_file.write('Query,RetrievedDocuments,Usage\n')
    random_file.write('Query,RetrievedDocuments\n')
    for image_id in ann_val['ImageID']:
        write_kaggle(ann_val,image_id,gt_file,ann_train,'Public')
        write_kaggle(ann_val,image_id,random_file,ann_train,'')
    for image_id in ann_test['ImageID']:
        write_kaggle(ann_test,image_id,gt_file,ann_train,'Private')
        write_kaggle(ann_test,image_id,random_file,ann_train,'')

    gt_file.close()
    random_file.close()

    print "Done. Annotation and random file saved"

if __name__ == "__main__":

    rootdir = sys.argv[1]
    # read annotation files
    ann_test = pandas.read_csv(os.path.join(rootdir,
                'test/annotation.txt'),sep = '\t')
    ann_val = pandas.read_csv(os.path.join(rootdir,
                'val/annotation.txt'),sep = '\t')
    ann_train = pandas.read_csv(os.path.join(rootdir,
                'train/annotation.txt'),sep = '\t')

    print 'test',len(ann_test)
    print 'val',len(ann_val),
    print 'train',len(ann_train)
    
    gt_file = open(os.path.join(rootdir,'kaggle_sol.csv'),'w')
    random_file = open(os.path.join(rootdir,'kaggle_rnd_bl.csv'),'w')
    convert_ranking_annotation(ann_test,ann_val,ann_train,gt_file,random_file)
