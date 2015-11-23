## Repositori de suport

Repositori de gestió de l'assignatura mitjançant l'eina d'issues.

### Suport
Plantegeu els vostres dubtes obrint un nou issue en aquest repositori.

### Instruccions per les entregues setmanals

Busqueu l'issue de l'entrega corresponent (El títol dels issues d'entrega és la data límit).

Responeu l'issue amb un missatge amb el nom del vostre equip i l'enllaç públic a la presentació a Google Drive.

### Instructions for the random classifier and ranker

- Edit the file `utils/params.py` so that `params['root']` points to the directory where you have the dataset `TerrassaBuildings900` and where you would like to store all the intermediate files.
- Run `utils/params.py`. Notice that this will create directories as well, so make sure you did the previous step ! 
- Run `build_database.py` to generate text files with image IDs for both the training and validation sets.
- Run `get_features.py` to generate random features for both training and validation sets and store them independently in dictionaries.
- Run `classify.py` and `rank.py` to generate results
- Run `eval_classification.py` and `eval_ranking.py` to evaluate the results. For a better analysis of the results, take a look at `notebooks/gdsa_s4.ipynb`. 


