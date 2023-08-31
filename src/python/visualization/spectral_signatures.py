### Visualizing spectral signatures from features: density distribution per band.
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap

# Input: ground truth data
ground_truth_ria_formosa = './data/processed/datasets/ria_formosa_truth_data2.csv'

# Output: density distribution figure
spectral_signature_ria_formosa_3class = './reports/spectral_signatures/ria_formosa_spectral_signatures_3class.png'
spectral_signature_ria_formosa_2class = './reports/spectral_signatures/ria_formosa_spectral_signatures_2class.png'

# RIA FORMOSA
truth_data = pd.read_csv(ground_truth_ria_formosa)


# SPECTRAL SIGNATURES FOR 3 CLASSES 

# Subset: only seagrass data
features = ['blue', 'green', 'red', 'nir', 'ndvi', 'dem']
subset = features
subset.append('seagrass_class')

truth_subset = truth_data[subset]

# Visualizing: creating figure with 6 subplots (1 per band)
fig_rows = 2
fig_cols = 3

subplots_indexes = []

for i in range(fig_rows):
    for j in range(fig_cols):
        subplots_indexes.append((i, j))

plt.close()
fig, axs = plt.subplots(nrows=fig_rows, ncols=fig_cols, figsize=(13.5, 6))

class_ids = np.unique(truth_subset['seagrass_class'])
colours = ['#C0C0C0', '#228B22', '#66CD00']  

sns.set_palette(sns.color_palette(colours))

for class_id in class_ids:
        for (i, j), band in zip(subplots_indexes, features):
                title = "{} band".format(str.capitalize(band))
                curr_data = truth_subset[truth_subset['seagrass_class']==class_id]
                sns.kdeplot(curr_data[band], fill=True, ax=axs[i,j]).set(xlabel=None, ylabel=None, title=title)

axs[1,0].set_title('NIR band')
axs[1,1].set_title('NDVI band')
axs[1,2].set_title('DEM band')

fig.subplots_adjust(left=0.087, right=0.845,top=0.940, bottom=0.110, hspace=0.3, wspace=0.5)
fig.legend(labels = ['Seagrass absence', 'Intertidal seagrass', 'Subtidal seagrass'], bbox_to_anchor=(0.99, 0.6))
fig.supxlabel('Values')
fig.supylabel('Density')
#plt.show()
plt.savefig(spectral_signature_ria_formosa_3class)

# ------

# SPECTRAL SIGNATURES FOR 2 CLASSES 
# Subset: only seagrass data
truth_sg = truth_data.loc[(truth_data['habitatclass'] == 'seagrass intertidal') | (truth_data['habitatclass'] == 'seagrass subtidal')]
features = ['blue', 'green', 'red', 'nir', 'ndvi', 'dem']
subset = features
subset.append('habitatclass')

truth_subset = truth_sg[subset]
truth_subset = truth_subset[truth_subset['dem'] > -100] # Exclude very small values

# Visualizing: creating figure with 6 subplots (1 per band)
fig_rows = 2
fig_cols = 3

subplots_indexes = []

for i in range(fig_rows):
    for j in range(fig_cols):
        subplots_indexes.append((i, j))

plt.close()
fig, axs = plt.subplots(nrows=fig_rows, ncols=fig_cols, figsize=(13.5, 6))

class_ids = np.unique(truth_subset['habitatclass'])
colours = ['#228B22', '#66CD00']  

sns.set_palette(sns.color_palette(colours))

for class_id in class_ids:
        for (i, j), band in zip(subplots_indexes, features):
                title = "{} band".format(str.capitalize(band))
                curr_data = truth_subset[truth_subset['habitatclass']==class_id]
                sns.kdeplot(curr_data[band], fill=True, ax=axs[i,j]).set(xlabel=None, ylabel=None, title=title)

axs[1,0].set_title('NIR band')
axs[1,1].set_title('NDVI band')
axs[1,2].set_title('DEM band')

fig.subplots_adjust(left=0.087, right=0.845,top=0.940, bottom=0.110, hspace=0.3, wspace=0.5)
fig.legend(labels = ['Intertidal seagrass', 'Subtidal seagrass'], bbox_to_anchor=(0.99, 0.6))
fig.supxlabel('Values')
fig.supylabel('Density')
#plt.show()
plt.savefig(spectral_signature_ria_formosa_2class)
