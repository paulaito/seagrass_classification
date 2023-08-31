# Hyperparameters tuning: Grid Search Cross-Validation

import pandas as pd
import rasterio as rio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score, make_scorer
import time
from splitting import X_train, y_train

time_0 = time.time()

rf = RandomForestClassifier(
    random_state=1)

# Define grid of hyperparameters
params_rf = {
        'n_estimators': [120, 200, 280, 400],
        'max_depth': [2, 6, 8, None],
        'min_samples_leaf': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5],
        'max_features': ['log2', 'sqrt']
        }
kappa_score = make_scorer(cohen_kappa_score)

## Perform CV with GridSearchCV
grid_rf = GridSearchCV(estimator=rf, 
                       param_grid=params_rf,
                       cv=5,
                       scoring=kappa_score,
                       verbose=1,
                       n_jobs=-1)


### 2: MODEL TRAINING (find best model hyperparameters)
grid_rf.fit(X_train, y_train)

## Get the best model from the hyperparameter tuning
best_model = grid_rf.best_estimator_
print(grid_rf.best_params_)
print(grid_rf.best_score_)
