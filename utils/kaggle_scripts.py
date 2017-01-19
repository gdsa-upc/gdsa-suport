import numpy as np
import pandas
import sys,os
import random
import copy

def write_kaggle(ann,image_id,file_to_save,ann_train,usage):

    i_class = list(ann.loc[ann['ImageID'] == image_id]['ClassID'])[0]
    file_to_save.write(image_id + ',')

    shuffled = copy.deepcopy(ann_train)
    if usage == '':
        #Here we are writing a sample ranking generated randomly
        to_write = shuffled['ImageID']
        random.shuffle(to_write)

        for i in to_write:
            file_to_save.write(i + ' ')
    else:
        # Here we write the ground truth ranking
        to_write = ann_train.loc[ann_train['ClassID'].isin([i_class])]['ImageID'].tolist()
        for i in to_write:
            file_to_save.write(i + ' ')

        if i_class == 'desconegut':
            usage = 'Ignored'
        file_to_save.write(','+usage)

    file_to_save.write('\n')

def rankings_csv(ann_test, ann_val, ann_train,gt_file,random_file):

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
    print 'val',len(ann_val)
    print 'train',len(ann_train)

    gt_file = open(os.path.join(rootdir,'kaggle_sol.csv'),'w')
    random_file = open(os.path.join(rootdir,'kaggle_rnd_bl.csv'),'w')
    rankings_csv(ann_test,ann_val,ann_train,gt_file,random_file)
