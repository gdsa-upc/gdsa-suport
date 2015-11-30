## Suport i entregues setmanals

### Eina d'issues
Plantegeu els vostres dubtes obrint un nou issue en aquest repositori.

### Instruccions per les entregues setmanals

Busqueu l'issue de l'entrega corresponent (El títol dels issues d'entrega és la data límit).

Responeu l'issue amb un missatge amb el nom del vostre equip i l'enllaç públic a la presentació a Google Drive.

## [NEW] Code Instructions

### Setup

- Edit the file `utils/params.py` so that `params['root']` points to the directory where you have the dataset `TerrassaBuildings900` and where you would like to store all the intermediate files. You can also modify some of the parameters there (such as the codebook size, the keypoint detector, etc.)
- Run `utils/params.py`. Notice that this will create directories as well, so make sure you did the previous step !
- Run `build_database.py` to generate text files with image IDs for both the training and validation sets. 

### Feature extraction

- Run `get_features.py`. This will first train a codebook, and then build BoW vectors for the training and validation/test images.

### S5: Ranking Instructions


- Run `rank.py` to compute distances from validation images to train images and generate the rankings.
- Run `eval_rankings.py` to obtain the mean Average Precision of your rankings.
- For a step by step tutorial and a better analysis of the results, check `notebooks/gdsa_s5.ipynb`

### S6: Classification Instructions

- To be updated

## Results

| Team                  | mAP (retrieval) | mAP (classification) |
| -------------         | --------------- | -------------------- |
| [Building Recognizer](http://gdsa-upc.github.io/Building-Recognizer/)   | TBD             | TBD                  |
| RdE                   | TBD             | TBD                  |
| [What a building !](http://gdsa-upc.github.io/What-a-building-App/)     | TBD             | TBD                  |
| Discover Terrassa     | TBD             | TBD                  |
| [Egara View](http://gdsa-upc.github.io/Egara-View/)            | TBD             | TBD                  |
| [International Team](http://gdsa-upc.github.io/International-Team/)    | TBD             | TBD                  |

