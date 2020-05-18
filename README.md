# DeepMet
Code for the paper "DeepMet: A Reading Comprehension Paradigm for Token-level Metaphor Detection".

## The Second Shared Task on Metaphor Detection
https://competitions.codalab.org/competitions/22188

## Project
	|--README.md
	|--corpora
	    |-- VUA # VUA corpora
	    |-- TOEFI # TOEFI is not a public data set. Please contact the competition organizer for data.
	    |-- MOH-X # MOH-X corpora
	    |-- TroFi # TroFi corpora
	|--data
	    |-- VUA # VUA data set
	    |-- TOEFI # TOEFI is not a public data set. Please contact the competition organizer for data.
	    |-- MOH-X # MOH-X data set
	    |-- TroFi # TroFi data set
	|-- log # Log files
	|-- model # Save models
	|-- output # Output files 
	|-- submit # Submit files
	|-- data_augmentation.py # Data augmentation
	|-- data_preprocessing.py # Get (context, question word, answer) triple data
	|-- DeepMet-toefi.py
	|-- DeepMet-vua.py
	|-- EDA.ipynb # Exploratory data analysis
    |-- ensemble_learning.py # ensemble learning
	|-- get_features.py # Get POS features
	|-- requirements.txt
	|-- submit_result.py # Submit result
	|-- toefi_preprocessing.py # TOEFI data preprocessing
	|-- vua_preprocessing.py # TVUA data preprocessing
	

## Environment
```
pip install -r requirements.txt
```

## Run
Using RoBERTa as the backbone network, we can get about 77 and 73.5 F1 scores on the VU verb and the all POS test set, respectively. Using Model ensemble strategy with different hyperparameters can increase F1 score by about 3%.

```
python DeepMet_vua.py
```
```
python DeepMet_toefi.py
```
```
python ensemble.py
```

## Citation
```
@inproceedings{su2020deepmet,
  title={DeepMet: A Reading Comprehension Paradigm for Token-level Metaphor Detection},
  author={Su, Chuandong and Fukumoto, Fumiyo and Huang, Xiaoxi and Wang, Rongbo and Chen, Zhiqun},
  booktitle={The Second Workshop on Figurative Language Processing},
  year={2020}
}
```
