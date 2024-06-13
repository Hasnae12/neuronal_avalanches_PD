#%%
"""
==============================================================
Group level analysis:

Steps:

- Read the 3D matrices of subjects (each subject has a shape of (n_channels, n_channels, n_avalanches)).
- Average the 3D matrices to obtain 2D transition matrices for subjects (each subject has a shape of (n_channels, n_channels)).
- Average across subjects for ON, average across subjects for OFF.
- Element-wise difference of the averages ON-OFF.

===============================================================

"""
 
# Authors:
# Hasnae AGOURAM   
# The functions of the avalanche transition matrix are carried out by Pierpaolo Sorrentino
# Date: 13/06/2024
# License: BSD (3-clause)

# Import libraries
import numpy as np
from matplotlib import pyplot as plt
import mne
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle


# STEP 1 : Read the 3D matrices of subjects (n_channels, n_channels, n_avalanches)
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']

loaded_data = {}
ATM_3D = {}

for subject in subjects:
    file_path = f'{subject}_ATM_3D.npz'
    loaded_data[subject] = np.load(file_path)
    ATM_3D[subject] = {'ON': loaded_data[subject]['array_on'], 'OFF': loaded_data[subject]['array_off']}

# STEP 2 : Average the 3D matrices to obtain 2D transition matrices for each subject (n_channels, n_channels)
    # While averaging along the third dimension, we don't consider the zero elements for both the observed and random differences.
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']

ATM_2D = {}

for subject in subjects:
    ATM_3D_ON = loaded_data[subject]['array_on']
    ATM_3D_OFF = loaded_data[subject]['array_off']
    
    ATM_2D[subject] = {
       'ON': np.sum(ATM_3D_ON, axis=2)/ np.sum(ATM_3D_ON != 0, axis=2),
       'OFF': np.sum (ATM_3D_OFF, axis=2)/ np.sum(ATM_3D_OFF != 0, axis=2)
    }
# Steps 3 : Observed difference:
"""
- Average across subjects for ON, average across subjects for OFF
- Element-wise difference of the averages ON-OFF
- Replicate the resulting matrix number of permutations times along the third dimension
"""
# Average across subjects for ON condition / Average across subjects for OFF condition
for subject in subjects:
    sum_ATM_ON = sum(ATM_2D[subject]['ON'] for subject in subjects)
    average_ATM_ON =  sum_ATM_ON/len(subjects)
    sum_ATM_OFF = sum(ATM_2D[subject]['OFF'] for subject in subjects)    
    average_ATM_OFF =  sum_ATM_OFF/len(subjects)
    
print(average_ATM_ON.shape, average_ATM_OFF.shape)   # The shape (n_channels, n_channels)
# Plot :Transition matrix: average across all subjects for MED ON/ average across all subjects for MED OFF
total_channels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4','F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
     
vmin = min(np.min(average_ATM_ON), np.min(average_ATM_OFF))
vmax = max(np.max(average_ATM_ON), np.max(average_ATM_OFF))  
print(vmin,vmax)
     
plt.subplot(1, 2, 1)
plt.imshow(average_ATM_ON, cmap='viridis', interpolation='none',vmin = vmin, vmax = vmax)
plt.colorbar(shrink=0.45)
row_labels = total_channels 
column_labels = total_channels
plt.xticks(np.arange(len(row_labels)), column_labels, fontsize='9', rotation='vertical')
plt.yticks(np.arange(len(row_labels)), row_labels, fontsize='9')
plt.title(f'Average across subjects for MED ON', fontsize =  '11')

plt.subplot(1, 2, 2)
plt.imshow(average_ATM_OFF, cmap='viridis', interpolation='none',vmin = vmin, vmax = vmax )
plt.colorbar(shrink=0.45)
plt.xticks(np.arange(len(row_labels)), column_labels, fontsize='9', rotation='vertical')
plt.yticks(np.arange(len(row_labels)), row_labels, fontsize='9')
plt.title(f'Average across subjects for MED OFF', fontsize =  '11')

plt.tight_layout()
plt.savefig('Average_subjects_ATMs' , dpi = 600)
plt.show()
# Element-wise difference of the averages ON-OFF
obs_diff = average_ATM_ON - average_ATM_OFF
total_channels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4','F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']

plt.imshow(obs_diff, cmap='viridis', interpolation='none')
plt.colorbar()

row_labels = total_channels 
column_labels = total_channels
plt.xticks(np.arange(len(column_labels)), column_labels, fontsize=9, rotation='vertical')
plt.yticks(np.arange(len(row_labels)), row_labels, fontsize=9, rotation='horizontal')   

plt.title('Element-wise difference ON-OFF of average transition matrices')
plt.savefig('Diff_Average_subjects_ATMs' , dpi = 600)
plt.show()

# Use a circular plot to display the average across subjects of ATMs ON
con= average_ATM_ON
node_names = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
node_colors = ['darkseagreen', 'darkseagreen', 'darkseagreen', 'darkseagreen','sandybrown', 'sandybrown', 'sandybrown', 'sandybrown','cyan', 'cyan', 'cyan', 'cyan','cyan', 'cyan']
node_order = ['R4', 'R3', 'R2', 'R1','L1', 'L2', 'L3', 'L4', 'F3', 'C3','Fz', 'Cz', 'F4', 'C4' ]
node_angles =  circular_layout(node_names, node_order, start_pos=0, start_between=False, group_boundaries=None, group_sep=10)
vmin = min(np.min(average_ATM_ON), np.min(average_ATM_OFF))
vmax = max(np.max(average_ATM_ON), np.max(average_ATM_OFF)) 
title =  "Average across subjects of ATMs ON"

fig, ax = plt.subplots(figsize=(8, 8), facecolor="white", subplot_kw=dict(polar=True))

plot_connectivity_circle(con, node_names, indices=None, n_lines=None, node_angles=node_angles, node_width=None, node_height=1, 
                         node_colors=node_colors, textcolor='black', facecolor='white', node_edgecolor='black', linewidth=3, 
                         colormap='YlOrBr', vmin=vmin, vmax=vmax, colorbar=True, title=title, colorbar_size=0.3, colorbar_pos=(-0.1, 0.1), 
                         fontsize_title=15, fontsize_names=16, fontsize_colorbar=12, padding=4.0, ax=ax, 
                         interactive=True, node_linewidth=2.5, show=True)

fig.tight_layout()
fname_fig = 'Average across subjects of ATMs ON.png'
fig.savefig(fname_fig, facecolor='white', dpi=600)

# Use a circular plot to display the average across subjects of ATMs OFF
con= average_ATM_OFF
vmin = min(np.min(average_ATM_ON), np.min(average_ATM_OFF))
vmax = max(np.max(average_ATM_ON), np.max(average_ATM_OFF)) 
title =  "Average across subjects of ATMs OFF"
fig, ax = plt.subplots(figsize=(8, 8), facecolor="white", subplot_kw=dict(polar=True))

plot_connectivity_circle(con, node_names, indices=None, n_lines=None, node_angles=node_angles, node_width=None, node_height=1, 
                         node_colors=node_colors, textcolor='black', facecolor='white', node_edgecolor='black', linewidth=3, 
                         colormap='YlOrBr', vmin=vmin, vmax=vmax, colorbar=True, title=title, colorbar_size=0.3, colorbar_pos=(-0.1, 0.1), 
                         fontsize_title=15, fontsize_names=16, fontsize_colorbar=12, padding=4.0, ax=ax, 
                         interactive=True, node_linewidth=2.5, show=True)

fig.tight_layout()
fname_fig = 'Average across subjects of ATMs OFF.png'
fig.savefig(fname_fig, facecolor='white', dpi=600)

# Use a circular plot to dislay the Element-wise difference of the averages ON-OFF
con= average_ATM_ON - average_ATM_OFF
vmin = np.min(obs_diff) 
vmax = np.max(obs_diff) 
title =  "Difference between the average ATMs ON and OFF"
fig, ax = plt.subplots(figsize=(8, 8), facecolor="white", subplot_kw=dict(polar=True))

plot_connectivity_circle(con, node_names, indices=None, n_lines=None, node_angles=node_angles, node_width=None, node_height=1, 
                         node_colors=node_colors, textcolor='black', facecolor='white', node_edgecolor='black', linewidth=3, 
                         colormap='YlOrBr', vmin=vmin, vmax=vmax, colorbar=True, title=title, colorbar_size=0.3, colorbar_pos=(-0.1, 0.1), 
                         fontsize_title=15, fontsize_names=16, fontsize_colorbar=12, padding=4.0, ax=ax, 
                         interactive=True, node_linewidth=2.5, show=True)

fig.tight_layout()
fname_fig = 'Difference between the average ATMs ON and OFF.png'
fig.savefig(fname_fig, facecolor='white', dpi=600)
