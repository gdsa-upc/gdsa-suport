import os
from params import get_params

def build_database(params):

    # List images
    image_names = os.listdir(os.path.join(params['root'],
                             params['database'],params['split'],'images'))

    # File to be saved
    file = open(os.path.join(params['root'],params['root_save'],
                             params['image_lists'],
                             params['split'] + '.txt'),'w')

    # Save image list to disk
    for imname in image_names:
        file.write(imname + "\n")
    file.close()

if __name__=="__main__":

    params = get_params()

    for split in ['train','val','test']:
        params['split'] = split
        build_database(params)
