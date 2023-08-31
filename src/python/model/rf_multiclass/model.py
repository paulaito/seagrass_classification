# Random Forest Classification

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from hyperparameters_tuning import best_model
from splitting import X_train, X_test, y_test

# Output paths
importances_rf_out = './reports/figures/feature_importance/ftimp_RF_seagrass_ria_formosa.csv'
confusion_matrix_out = './reports/confusion_matrix/ria_formosa/cm_ria_formosa_seagrass.png'

### MODEL VALIDATION
y_pred = best_model.predict(X_test)

## Score metrics

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision, recall, f-score
precision_recall_fscore = precision_recall_fscore_support(y_test, y_pred)

precision = precision_recall_fscore[0]
recall = precision_recall_fscore[1]
f1 =  precision_recall_fscore[2]

print(accuracy, precision, recall, f1)

# Feature Importance
importances_rf = pd.Series(best_model.feature_importances_, index = X_train.columns)
importances_rf = importances_rf.sort_values()
importances_rf.to_csv(importances_rf_out)

# Confusion matrix
cm = confusion_matrix(y_test,best_model.predict(X_test))

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted classes')
plt.ylabel('True classes')
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
plt.xticks([0.5, 1.5, 2.5], labels=['Absence', 'Intertidal seagrass', 'Subtidal seagrass'])
plt.yticks([0.5, 1.5, 2.5], labels=['Absence', 'Intertidal seagrass', 'Subtidal seagrass'])
plt.show()

fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted Class', size=12)
ax.set_ylabel('True Class', size=12)
ax.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
ax.set_xticks([0.5, 1.5, 2.5], labels=['Absence', 'Intertidal seagrass', 'Subtidal seagrass'])
ax.set_yticks([0.5, 1.5, 2.5], labels=['Absence', 'Intertidal seagrass', 'Subtidal seagrass'])
ax.xaxis.set_label_position('top')
ax.xaxis.set_label_coords(.5, 1.12)
ax.yaxis.set_label_coords(-.1, .5)
#plt.show()
plt.savefig(confusion_matrix_out)

# Cohen Kappa Score
cohen_kappa = cohen_kappa_score(y_test, y_pred)
print(cohen_kappa)
##################################
