# Split into train and test
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

### 1: READ GROUND TRUTH DATA
ria_ground_truth_dataset = './data/processed/datasets/ria_formosa_truth_data.csv'

ria_truth_data = pd.read_csv(ria_ground_truth_dataset)

##
### 2: PREPARING DATASET

## Define features and labels
features = ['blue', 'green', 'red', 'nir', 'ndvi', 'dem']

X_ria = ria_truth_data[features]
y_ria = ria_truth_data['seagrass_class']


## Split into 80% train, 20% test in stratified sampling (keep proportion of classes)
X_train, X_test, y_train, y_test = train_test_split(X_ria, y_ria, stratify=y_ria, test_size = 0.2, random_state=1)


