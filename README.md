## Support and Weekly deliverables

### Instructions for students:

- Issues: Open new issues in this repository to pose your questions.
- Deliverables: A new issue will be created in this repository for you to upload your weekly deliverables (Google Slides). Find it and answer with a public link to your slides before the deadline.

## Code Instructions

### Requirements
- [Python 2.7](https://www.python.org/download/releases/2.7/)
- Install pip following [these](https://pip.pypa.io/en/stable/installing/) instructions.
- Run ```pip install -r requirements.txt``` to install code dependencies.
- Install OpenCV 2.4.x from [source](http://opencv.org/downloads.html) (instructions [here](http://docs.opencv.org/2.4/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html)) or using the [package manager](https://www.enthought.com/products/canopy/package-index/) in Canopy with the free academic subscription (the latter is recommended for simplicity).

Note: version 3.x of OpenCV will not work with the current version of the code since SIFT and SURF extraction modules were removed.
### Setup

- Edit the file `src/params.py` so that `params['root']` points to the directory where you have the dataset `TerrassaBuildings900` and where you would like to store all the intermediate files. You can also modify some of the parameters there (such as the codebook size, the keypoint detector, etc.)
- Run `src/params.py`. Notice that this will create directories as well, so make sure you did the previous step !
- Run `src/build_database.py` to generate text files with image IDs for both the training and validation sets. 

### Feature extraction

- Run `src/get_features.py`. This will first train a codebook, and then build BoW vectors for the training and validation/test images.

### S5: Ranking Instructions

- Run `src/rank.py` to compute distances from validation images to train images and generate the rankings.
- Run `src/eval_rankings.py` to obtain the mean Average Precision of your rankings.
- For a step by step tutorial and a better analysis of the results, check `notebooks/gdsa_s5.ipynb`

### S6: Classification Instructions

- Run `src/classify.py`. This will train a classifier and use it to predict classes for validation images.
- Run `src/eval_classification.py` to get evaluation metrics of your classifier predictions.
- Check `notebooks/gdsa_s6.ipynb` to see a tested example.

### Kaggle rankings

There is a script ```utils/save_for_kaggle.py ``` that will take your rankings and convert them to be uploaded to Kaggle.

### Deep features (coming soon !)

The script ```src/deepfeats.py``` extracts VGG-16 features using Keras and generates rankings in Kaggle format. It assumes you have first run ```src/build_database.py```. This is an alternative to the feature extraction pipeline that relies on local features (e.g. SIFT)
