#%%
"""
==============================================================
 Subject-level analysis: Statistical pipeline used to identify the edges that exhibit significant differences between the ON-levodopa and OFF-levodopa conditions at the subject level.
===============================================================
Steps :

-Read the 3D transition matrices for ON and OFF of each subject (the shape for each subject and each condition is (nchannels, nchannels,number_of_avalanches)).

Step1: Observed difference:
    - Average of the 3D matrices along the third dimension (number of avalanches) for each condition ON/OFF  for each subject.
    - Element-wise absolute difference of the averages ON-OFF for each subject.
    - Replicate the resulting matrix number of permutations times along the third dimension.

Step2: Random differences:
    - Concatenate the transition matrices from both ON and OFF (n_channels, n_channels, total_number_avalanches) for each subject.
    - Perform a permutation test by randomly mixing avalanche-specific transition matrices from ON and OFF conditions for each subject.

Step3: Compare the random and observed differences:
    - Calculate the proportion of random differences that are greater than the observed difference for each subject.
    
===============================================================

"""
# Authors:
# Hasnae AGOURAM   
# The functions of the avalanche transition matrix are carried out by Pierpaolo Sorrentino
# The percolation code was done in collaboration with Matteo Neri
# Date: 13/06/2024
# License: BSD (3-clause)

 
# Import libraries
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 
from mne.stats import fdr_correction
import mne
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle
from matplotlib.ticker import PercentFormatter

# Read the 3D matrices :(n_channels, n_channels,number_of_avalanches) for each subject
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']

loaded_data = {}
ATM_3D = {}

for subject in subjects:
    file_path = f'{subject}_ATM_3D.npz'
    loaded_data[subject] = np.load(file_path)
    ATM_3D[subject] = {'ON': loaded_data[subject]['array_on'], 'OFF': loaded_data[subject]['array_off']}

"""
Step 1 : Observed difference :

1) - Average of the 3D matrices along the third dimension (number of avalanches) for each condition ON/OFF for each subject.
2) - Element-wise absolute difference of the averages ON-OFF for each subject.
3) - Replicate the resulting matrix number of permutations times along the third dimension.
"""

# 1) Average of transition matrices along the third dimension (number of avalanches) for each subject and condition
    # While averaging along the third dimension, we don't consider the zero elements for both the observed and random differences.
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']

ATM_2D = {}

for subject in subjects:
    ATM_3D_ON = loaded_data[subject]['array_on']
    ATM_3D_OFF = loaded_data[subject]['array_off']

   
   
    ATM_2D[subject] = {
        'ON': np.sum(ATM_3D_ON, axis=2) / np.sum(ATM_3D_ON != 0, axis=2),
        'OFF': np.sum(ATM_3D_OFF, axis=2) / np.sum(ATM_3D_OFF != 0, axis=2)
    }

# The shape of the averaged transition matrices is (n_channels, n_channels) for each subject and each condition

# 2) Compute the element-wise absolute difference of the 2D transition matrices 'ON' and 'OFF' for each subject
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']

obs_diff_dict = {}

for subject in subjects:
    obs_diff_dict[subject] = np.abs(ATM_2D[subject]['ON'] - ATM_2D[subject]['OFF'])
    print(f"obs_diff_{subject} shape: {obs_diff_dict[subject].shape}")

# 3) Replicate the resulting matrix (element-wise absolute difference matrix) a number of permutations times along the third dimension for each subject.
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']
num_perm=10000
rep_obs_dict = {}

for subject in subjects:
    
# Observed difference: Element-wise absolute difference between the two matrices ON and OFF for each subject
    obs_diff_dict[subject] = np.abs(ATM_2D[subject]['ON'] - ATM_2D[subject]['OFF'])
    
# Replication of obs_diff number of permutation times along the third dimension for each subject
    rep_obs_dict[subject] = np.tile(obs_diff_dict[subject][:, :, np.newaxis], (1, 1, num_perm))
    print(f"obs_diff_{subject} shape: {rep_obs_dict[subject].shape}") # shape (n_channels, n_channels, num_perm)

"""
Step 2 & Step 3: Compute random differences and compare them with observed differences:

    1) - Concatenate the transition matrices from both ON and OFF (n_channels, n_channels, total_number_avalanches) for each subject.
    2) - Perform a permutation test by randomly mixing avalanche-specific transition matrices from ON and OFF conditions for each subject.
    3) - Calculate the proportion of random differences that are greater than the observed difference for each subject.
"""

# 1) Concatenate the 3D transition matrices from both 'ON' and 'OFF' for each subject
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']
concatenated_matrices = []

for subject in subjects:
    concatenated_matrices.append(np.concatenate([ATM_3D[subject]['ON'], ATM_3D[subject]['OFF']], axis=2))
    print(f"Shape for {subject}: {concatenated_matrices[-1].shape}")
"""
2) Perform a permutation test
3)Calculate the proportion of random differences that are greater than the observed difference for each subject
"""
import random
results = {}
random_observed_diffs = {}
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']
num_perm = 10000 # Number of permutation

for subject in subjects:

# Concatenate the 3D transition matrices from both 'ON' and 'OFF' for each subject 
    TMs = np.concatenate([ATM_3D[subject]['ON'], ATM_3D[subject]['OFF']], axis=2)
# Number of channels
    N_chan = TMs.shape[0]  
# The number of avalanches in 'ON' condition for each subject
    n_avalanches_on = len(ATM_3D[subject]['ON'][0, 0, :])  

# Random differences
    rand_diffs = np.zeros((N_chan, N_chan, num_perm))
    def group_generator(x):
        return random.sample(x, len(x))
    
# Permutation test 
    for kk1 in range(num_perm):
        rand_temp = group_generator(list(range(TMs.shape[2])))   
        temp_TM = TMs[:, :, rand_temp]

        rand1 = np.sum(temp_TM[:, :, :n_avalanches_on + 1], axis=2) / np.sum(
           temp_TM[:, :, :n_avalanches_on + 1] != 0, axis=2)
        rand2 = np.sum(temp_TM[:, :, n_avalanches_on + 1:], axis=2) / np.sum(
           temp_TM[:, :, n_avalanches_on + 1:] != 0, axis=2)

        
        rand1[np.isnan(rand1)] = 0
        rand2[np.isnan(rand2)] = 0
        
# The shape of Random differences(n_channels, n_channels, num_perm)
        rand_diffs[:, :, kk1] = np.abs(rand1 - rand2) 
    
# Observed difference: Element-wise absolute difference between the two matrices ON and OFF for each subject
    obs_diff = np.abs(ATM_2D[subject]['ON'] - ATM_2D[subject]['OFF'])

# Replication of obs_diff number of permutation times along the third dimension
    rep_obs = np.tile(obs_diff[:, :, np.newaxis], (1, 1, num_perm)) #shape (n_channels, n_channels, num_perm)

# Calculate the proportion of random differences greater than the observed difference for each subject
    jj = np.sum(rand_diffs > rep_obs, axis=2) / num_perm
    
# Matrix of p-values for each subject
    results[subject] = jj
    
# Random and observed differences for each subject
    random_observed_diffs [subject] = {
        'Random_differences':  rand_diffs,
        'observed_differences':  rep_obs
    }

# Additional analyses

# 1) Distribution of Null Hypothesis with Observed Difference: One edge for each subject
for subject in subjects :
# The random differences for one edge
    sns.histplot(random_observed_diffs[subject]['Random_differences'][5,13,:] , kde=True, label='Null Hypothesis', color='blue')

# The observed values for one edge 
    plt.axvline(random_observed_diffs[subject]['observed_differences'][5,13,0], color='red', linestyle='dashed', label='Observed Difference')
    plt.text(0.025, 150, f'p-value: {results[subject][5,13]:.4f}', color='black')
    plt.xlabel('Difference Values', fontsize = 15)
    plt.ylabel('Density', fontsize = 15)
    plt.title(f'Distribution of Null Hypothesis with Observed Difference: for {subject} and one edge')
    plt.legend()
    #plt.savefig("Distribution_permutation_test_"+str(subject)+".jpg", dpi=600)
    plt.show()

# 2)- The resulting matrix of p-values for each subject represents the proportion of random differences greater than the observed difference
    # Plot the significant egdes : if p_values < 0.05: result = 1(reject the null hypothesis)
for subject in subjects:
    plt.imshow(np.where(results[subject]<0.05,1,0), cmap='viridis', interpolation='none', vmin=0, vmax=1)
    plt.colorbar(shrink=0.45)
    row_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    column_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    plt.xticks(np.arange(len(row_labels)), column_labels, fontsize='9', rotation='vertical')
    plt.yticks(np.arange(len(row_labels)), row_labels, fontsize='9')
    plt.title(f'Significant P values for {subject}')
    plt.tight_layout()
    plt.xlabel('channels')
    plt.ylabel('channels')
    plt.show()
#3)- Correction for multiple comparisons using False Discovery Rate (FDR) on matrices of p-values for each subject.
    # Plot the Significant FDR-Corrected edges: if p_values < 0.05: result = 1(reject the null hypothesis)

subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']
reject_fdr = {}
pval_fdr = {}

for subject in subjects:
    reject_fdr[subject], pval_fdr[subject] = fdr_correction(results[subject], alpha=0.05, method="indep")
    plt.imshow(np.where(pval_fdr[subject]<0.05,1,0), cmap='viridis', interpolation='none')
    plt.title(f'Significant FDR-Corrected p-values for {subject}')
    row_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    column_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    plt.xticks(np.arange(len(row_labels)), column_labels, fontsize='9', rotation='vertical')
    plt.yticks(np.arange(len(row_labels)), row_labels, fontsize='9')
    cbar = plt.colorbar()
    cbar.set_label('Significant FDR-Corrected p-values')
    plt.xlabel('channels')
    plt.ylabel('channels')
    plt.show()
# 4)- Direction of significant edges after FDR correction: either 'ON > OFF' or 'OFF > ON' for each subject
    # The resulting matrices contain three different values: ON greater than OFF (value=1), ON lower than OFF (value=-1), or non-significant edges (value=0)
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']

for subject in subjects:
    direction_matrix = np.sign(ATM_2D[subject]['ON'] - ATM_2D[subject]['OFF'])
    significant_matrix = np.where(pval_fdr[subject] < 0.05, 1, 0)

# Multiply the direction of the difference of the transition matrix (ON-OFF) by the significant matrix(p-values)
    result_direction = direction_matrix * significant_matrix

    plt.imshow(result_direction, cmap='viridis', interpolation='none')
    cbar = plt.colorbar(shrink = 0.9)
    cbar.set_label('Direction of significant p values after FDR correction')
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['-1', '0', '1'], fontsize='15')

    row_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    column_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']

    plt.xticks(np.arange(len(row_labels)), column_labels, fontsize='15', rotation='vertical')
    plt.yticks(np.arange(len(row_labels)), row_labels, fontsize='15')

    plt.xlabel('channels', fontsize='15')
    plt.ylabel('channels', fontsize='15')
    plt.tight_layout()
    plt.savefig('significant_edges_FDR'+ str(subject) , dpi = 600)
    plt.show()

# 5)- Consistently significant edges across subjects after FDR correction: where ON > OFF vs. ON < OFF
    # By summing the matrices of significant edges where 'ON > OFF' versus 'OFF > ON' across subjects 

subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']

result_direction_fdr = {}
result_direction_on_gt_off_fdr = {}
result_direction_off_gt_on_fdr = {}
reject_fdr = {}
pval_fdr = {}

total_sum_matrix_on_gt_off_fdr = np.zeros_like(ATM_2D[subjects[0]]['ON'])
total_sum_matrix_off_gt_on_fdr = np.zeros_like(ATM_2D[subjects[0]]['ON'])

for subject in subjects:
    reject_fdr[subject], pval_fdr[subject] = fdr_correction(results[subject], alpha=0.05, method="indep")
    
# Multiply the direction of the difference of the transition matrix (ON-OFF) by the Significant FDR-Corrected p-values
 
    result_direction_fdr = np.sign(ATM_2D[subject]['ON'] - ATM_2D[subject]['OFF']) * np.where(pval_fdr[subject]<0.05,1,0)
    
# Significant edges: ON greater than OFF 
    result_direction_on_gt_off_fdr[subject] = np.where(result_direction_fdr == 1, result_direction_fdr, 0)
    
# Significant edges: OFF greater than ON 
    result_direction_off_gt_on_fdr[subject] = np.abs(np.where(result_direction_fdr == -1, result_direction_fdr, 0)) #Absolute values are important

# Sum the matrices of significant edges where 'ON > OFF' versus 'OFF > ON' across subjects 
    total_sum_matrix_on_gt_off_fdr += result_direction_on_gt_off_fdr[subject]
    total_sum_matrix_off_gt_on_fdr += result_direction_off_gt_on_fdr[subject]

print(total_sum_matrix_on_gt_off_fdr.shape, total_sum_matrix_off_gt_on_fdr.shape)

# PLot the consistently significant edges across subjects after FDR correction: where ON > OFF vs. ON < OFF
vmin= min(np.min(total_sum_matrix_on_gt_off_fdr), np.min(total_sum_matrix_off_gt_on_fdr))
vmax= max(np.max(total_sum_matrix_on_gt_off_fdr), np.max(total_sum_matrix_off_gt_on_fdr))
 
plt.subplot(1, 2, 1)
im1 = plt.imshow(total_sum_matrix_on_gt_off_fdr, cmap='viridis', interpolation='none', vmin=vmin, vmax=vmax)
plt.xticks(np.arange(len(row_labels)), column_labels, fontsize='9', rotation='vertical')
plt.yticks(np.arange(len(row_labels)), row_labels, fontsize='9')
cbar1 = plt.colorbar(im1, shrink=0.45)
cbar1.set_label('Number of subjects', fontsize='9')
plt.xlabel('channels', fontsize='9')
plt.ylabel('channels', fontsize='9')
plt.title('ON>OFF', fontsize='12')

plt.subplot(1, 2, 2)
im2 = plt.imshow(total_sum_matrix_off_gt_on_fdr, cmap='viridis', interpolation='none', vmin=vmin, vmax=vmax)
plt.xticks(np.arange(len(row_labels)), column_labels, fontsize='9', rotation='vertical')
plt.yticks(np.arange(len(row_labels)), row_labels, fontsize='9')
cbar2 = plt.colorbar(im2, shrink=0.45)
cbar2.set_label('Number of subjects', fontsize='9')
plt.xlabel('channels', fontsize='9')
plt.ylabel('channels', fontsize='9')
plt.title('OFF>ON', fontsize='12')


plt.suptitle("Consistently significant edges across subjects after FDR correction: 'ON > OFF' vs. 'OFF > ON'")
plt.tight_layout()
plt.savefig('Consistently_significant_edges_FDR' , dpi = 600)

plt.show()

# Use a circular plot to display the consistency significant edges across all subjects for ON>OFF
con= total_sum_matrix_on_gt_off_fdr 
node_names = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
node_colors = ['darkseagreen', 'darkseagreen', 'darkseagreen', 'darkseagreen','sandybrown', 'sandybrown', 'sandybrown', 'sandybrown','cyan', 'cyan', 'cyan', 'cyan','cyan', 'cyan']
node_order = ['R4', 'R3', 'R2', 'R1','L1', 'L2', 'L3', 'L4', 'F3', 'C3','Fz', 'Cz', 'F4', 'C4' ]
node_angles =  circular_layout(node_names, node_order, start_pos=0, start_between=False, group_boundaries=None, group_sep=10)
vmin= min(np.min(total_sum_matrix_on_gt_off_fdr), np.min(total_sum_matrix_off_gt_on_fdr))
vmax= max(np.max(total_sum_matrix_on_gt_off_fdr), np.max(total_sum_matrix_off_gt_on_fdr))
colormap= 'hot'
title =  "Consistently significant edges across all patients in the ON>OFF"
fig, ax = plt.subplots(figsize=(7, 7), facecolor="white", subplot_kw=dict(polar=True))

plot_connectivity_circle(con, node_names, indices=None, n_lines=None, node_angles=node_angles, node_width=None, node_height=1, 
                         node_colors=node_colors, textcolor='black', facecolor='white', 
                         node_edgecolor='black', linewidth=1, 
                         colormap='YlOrBr', vmin= vmin, vmax=vmax, colorbar=True, title=title, colorbar_size=0.3, colorbar_pos=(-0.1, 0.1), 
                         fontsize_title=10, fontsize_names=10, fontsize_colorbar=12, padding=2.0, ax=ax, 
                         interactive=True, node_linewidth=1, show=True)

fig.tight_layout()
fname_fig = 'Consistently significant edges across all patients in the ON greater than OFF.png'
fig.savefig(fname_fig, facecolor='white', dpi=600)

# Use a circular plot to display the consistency significant edges across all subjects for OFF>ON

con= total_sum_matrix_off_gt_on_fdr 
node_angles =  circular_layout(node_names, node_order, start_pos=0, start_between=False, group_boundaries=None, group_sep=10)
vmin= min(np.min(total_sum_matrix_on_gt_off_fdr), np.min(total_sum_matrix_off_gt_on_fdr))
vmax= max(np.max(total_sum_matrix_on_gt_off_fdr), np.max(total_sum_matrix_off_gt_on_fdr))
title =  "Consistently significant edges across all patients in the OFF>ON"

fig, ax = plt.subplots(figsize=(7, 7), facecolor="white", subplot_kw=dict(polar=True))

plot_connectivity_circle(con, node_names, indices=None, n_lines=None, node_angles=node_angles, node_width=None, node_height=1, 
                         node_colors=node_colors, textcolor='black', facecolor='white', 
                         node_edgecolor='black', linewidth=1, 
                         colormap='YlOrBr', vmin= vmin, vmax=vmax, colorbar=True, title=title, colorbar_size=0.3, colorbar_pos=(-0.1, 0.1), 
                         fontsize_title=10, fontsize_names=10, fontsize_colorbar=12, padding=2.0, ax=ax, 
                         interactive=True, node_linewidth=1, show=True)

fig.tight_layout()
fname_fig = 'Consistently significant edges across all patients in the OFF greater than ON.png'
fig.savefig(fname_fig, facecolor='white', dpi=600)

# Use a circular plot to display the consistency significant edges for more than 9 subjects: where ON > OFF
con= np.where(total_sum_matrix_on_gt_off_fdr  >=  9, 1, 0)   
title =  "Consistently significant edges for more than 9 patients in the ON-levodopa state (after BH correction)"

fig, ax = plt.subplots(figsize=(6, 6), facecolor="white", subplot_kw=dict(polar=True))

plot_connectivity_circle(con, node_names, indices=None, n_lines=None, node_angles=node_angles, node_width=None, node_height=1, 
                         node_colors=node_colors, textcolor='black', facecolor='white', 
                         node_edgecolor='black', linewidth=1, 
                         colormap='binary', vmin= None, vmax=None, colorbar=False, title=title, colorbar_size=0.3, colorbar_pos=(-0.1, 0.1),
                         fontsize_title=10, fontsize_names=10, padding=2.0, ax=ax,
                         interactive=True, node_linewidth=1, show=True)

fig.tight_layout()
fname_fig = 'Consistently significant edges for more than 9 patients in the ON-levodopa state-after BH correction.png'
fig.savefig(fname_fig, facecolor='white', dpi=600)

# 6)-The number of significant edges for at least n patients, 1<n<11 in both cases: ON>OFF, and OFF>ON.
List_percolated_on_gt_off = []
List_percolated_off_gt_on = []

for i in np.arange(1, 12)[::-1]:
    List_percolated_on_gt_off.append(np.sum(total_sum_matrix_on_gt_off_fdr >= i))
    List_percolated_off_gt_on.append(np.sum(total_sum_matrix_off_gt_on_fdr >= i))
plt.figure ( figsize = (5,5) )
plt.plot([(x / 196) * 100 for x in List_percolated_on_gt_off], label='ON>OFF',linewidth=2 ,color='blue', alpha=0.8)
plt.plot([(x / 196) * 100 for x in List_percolated_off_gt_on], label='OFF>ON',linewidth=2,color='orange', alpha=0.8)
plt.xticks(np.arange(0, 11), np.arange(1, 12)[::-1], fontsize=12,fontweight = 'semibold')
plt.yticks(fontsize=12,fontweight = 'semibold')
plt.xlabel('Number of patients', fontsize=15,fontweight = 'bold')
plt.ylabel('% of edges', fontsize=15,fontweight = 'bold')
plt.title('The number of significant edges for at least n patients, 1 < n < 11')
plt.legend(fontsize='12', loc='upper left')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.tight_layout()

for pos in ['right', 'top']: 
    plt.gca().spines[pos].set_visible(False) 

plt.grid()
plt.savefig("The number of significant edges for at least n patients", dpi=600)

plt.show()

