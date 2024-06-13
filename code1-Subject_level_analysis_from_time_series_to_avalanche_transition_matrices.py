#%%
"""
==============================================================
 To estimate the neuronal avalanches analysis on the epoched and preprocessed resting-state dataset of 11 Parkinsonian patients under two conditions (medication ON and OFF)
===============================================================
Steps:

To perform the spread of avalanches analysis on the epoched and preprocessed resting-state dataset of 11 Parkinsonian patients under two conditions (medication ON and OFF), I follow these steps:

- Read the epoched data.
- Concatenate the data (to convert it into continuous data).
- Z-score the continuous data.
- Epoch the z-scored continuous data.
- Binarize the z-scored epoched data (using a threshold).
- Compute the transition matrices.

 * Remark: I concatenate the epochs before z-scoring because z-scoring the epoched data introduces noise (edge effect). Then, I re-epoch the data because our preprocessed data is epoched data (after rejecting noisy segments). Afterward, I follow the procedure to compute the transition matrices using the z-scored epoched data.

===============================================================
"""
# Authors:
# Hasnae AGOURAM   
# The functions of the avalanche transition matrix are carried out by Pierpaolo Sorrentino
# Date: 13/06/2024
# License: BSD (3-clause)

# Import libraries
from matplotlib import pyplot as plt
from mne import create_info
import numpy as np
import mne,glob
import scipy.io 
from scipy import stats
import mat73


# Part 1 : Data of Medication OFF
"""
-1)  Read the epoched data.
-2)  Concatenate the data (to convert it into continuous data).
-3)  Z-score the continuous data.
-4) Epoch the z-scored continuous data.
-5) Reshape the epoched and z-scored time series to (trials, times, channels).
"""
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']
continuous_off = {}
transpose_off = {}
z_continuous_off = {}
raw_off = {}
event_array = {}
epochs = {}
event_id = {'event_type': 1} 
tmin, tmax = 0, 4
sfreq = 512
original_ch_names = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'Fz', 'F4', 'C3', 'C4', 'Cz']
new_ch_names = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
ch_types = ['eeg'] * len(original_ch_names)  


sampling_infos = {}

for subject in subjects:
    
#Read the preprocessed and epoched data
    info_original = mne.create_info(original_ch_names, sfreq, ch_types=ch_types, verbose=None)
    med_off_file_path = f"/home/schweyantoine/Bureau/Data/RESTING-DATA/preprocessed_data/dataClean_step1/dataClean_step1_{subject}_MED_OFF.mat"
    epochs_off = mne.io.read_epochs_fieldtrip(med_off_file_path, info_original, data_name='dataClean_step1', trialinfo_column=6)
    
#Rearrange the channel order to facilitate interpretation later on with the transition matrices
    channel_mapping = [original_ch_names.index(ch) for ch in new_ch_names]
    rearranged_data = epochs_off.get_data()[:, channel_mapping, :]
    epochs[subject] = {'OFF': rearranged_data}

# Concatenate and transpose the epoched data to obtain continuous data
    continuous_off[subject] = np.concatenate(epochs[subject]['OFF'], axis=1)
    transpose_off[subject] = np.transpose(continuous_off[subject], (1, 0))
    
# Z-score the continuous data 
    z_continuous_off[subject] = stats.zscore(transpose_off[subject], axis=0)
    z_continuous_off[subject] = np.transpose(z_continuous_off[subject], (1, 0))

# Epoch the z-scored continuous data
    info_new = mne.create_info(new_ch_names, sfreq, ch_types)
    raw_off[subject] = mne.io.RawArray(z_continuous_off[subject], info_new)

# Read the sampling info
    med_off = f"/home/schweyantoine/Bureau/Data/RESTING-DATA/preprocessed_data/dataClean_step1/dataClean_step1_{subject}_MED_OFF.mat"
    mat_med_off = mat73.loadmat(med_off)
    sampling_infos[subject] = mat_med_off['dataClean_step1']['sampleinfo']

# Convert the sampling info to an event array
    event_array[subject] = np.column_stack(
        (sampling_infos[subject][:, 0].astype(int), np.zeros(len(sampling_infos[subject]), dtype=int),
         np.ones(len(sampling_infos[subject]), dtype=int)))

# Z-score the epoched data with MNE function
    epochs[subject] = mne.Epochs(raw_off[subject], events=event_array[subject], event_id=event_id, tmin=tmin,
                                  tmax=tmax, baseline=None, preload=True)

# Reshape the epoched and z-scored time series to (trials, times, channels).
zscore_epoched_time_series_off = {}
for subject in subjects : 

    zscore_epoched_time_series_off [subject]=  np.transpose(epochs[subject].get_data(copy=True), (0, 2, 1))
    print (zscore_epoched_time_series_off[subject].shape)

#Functions
def threshold_mat(data,thresh=2):
    current_data=data
    binarized_data=np.where(np.abs(current_data)>thresh,1,0)
    return (binarized_data)

def transprob(aval,nregions): # (t,r)
    mat = np.zeros((nregions, nregions))
    norm = np.sum(aval, axis=0)
    for t in range(len(aval) - 1):
        ini = np.where(aval[t] == 1)
        mat[ini] += aval[t + 1]
    mat[norm != 0] = mat[norm != 0] / norm[norm != 0][:, None]
    return mat

def Transprob(ZBIN,nregions, val_duration): # (t,r)
    mat = np.zeros((nregions, nregions))
    A = np.sum(ZBIN, axis=1)
    a = np.arange(len(ZBIN))
    idx = np.where(A != 0)[0]
    aout = np.split(a[idx], np.where(np.diff(idx) != 1)[0] + 1)
    ifi = 0
    for iaut in range(len(aout)):
        if len(aout[iaut]) > val_duration:
            mat += transprob(ZBIN[aout[iaut]],nregions)
            ifi += 1
    mat = mat / ifi
    return mat,aout
    
def find_avalanches(data,thresh=2, val_duration=2):
    binarized_data=threshold_mat(data,thresh=thresh)
    N=binarized_data.shape[1]
    mat, aout = Transprob(binarized_data,N,val_duration)
    aout=np.array(aout,dtype=object)
    list_length=[len(i) for i in aout]
    unique_sizes=set(list_length)
    min_size,max_size=min(list_length),max(list_length)
    list_avalanches_bysize={i:[] for i in unique_sizes}
    for s in aout:
        n=len(s)
        list_avalanches_bysize[n].append(s)
    return(aout,min_size,max_size,list_avalanches_bysize, mat)
    
# Compute avalanches for each subject in the medication OFF condition
    #Compute transition matrices for each subject; shape is (n_channels, n_channels, n_avalanches)
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']


transition_matrices_dict_avalanches_off = {}


for subject in subjects:
     
    ntrials, ntimes, n_channels = zscore_epoched_time_series_off[subject].shape
    
    transition_matrices_off_list = []

    for trial in range(ntrials):
        avalanches, _, _, _, mat = find_avalanches(
            zscore_epoched_time_series_off[subject][trial, :, :], thresh=2, val_duration=2)
        
        # Each avalanche has a transition matrix of dimensions (nchannels, nchannels)
        for avalanche in avalanches:
            transition_matrices_off_list.append(mat)
            
    # Each subject has a 3D transition matrix (n channels, n channels, total number of avalanches)
    transition_matrices_off = np.stack(transition_matrices_off_list, axis=2)


    transition_matrices_dict_avalanches_off[subject] = transition_matrices_off
    print(transition_matrices_dict_avalanches_off[subject].shape)
    
# Plot the average transition matrix for each subject  
"""    
Because I added vmin and vmax to be computed for each subject between the average transition matrices for ON and OFF conditions, there is a need to run the ATMs for the ON conditions before running this snippet of code or comment the vmin and vmax.
"""
vmin = {}
vmax = {}

num_subjects = len(subjects)
num_rows = (num_subjects // 3) + (num_subjects % 3 > 0)
num_cols = min(num_subjects, 3)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

for i, subject in enumerate(subjects):
    row, col = divmod(i, 3)  
    ax = axes[row, col] if num_subjects > 1 else axes  
    # Vmin and Vmax
    vmin[subject] = min(np.min(transition_matrices_dict_avalanches_off[subject].mean(axis=2)), np.min(transition_matrices_dict_avalanches_on[subject].mean(axis=2)))
    vmax[subject] = max(np.max(transition_matrices_dict_avalanches_off[subject].mean(axis=2)), np.max(transition_matrices_dict_avalanches_on[subject].mean(axis=2))) 
    # Plot the transition matrix: average across avalanches
    im = ax.imshow(transition_matrices_dict_avalanches_off[subject].mean(axis=2), cmap='viridis', interpolation='none')#, vmin=vmin[subject], vmax=vmax[subject])
    ax.set_title(f'Transition matrix for {subject} MED OFF')
    fig.colorbar(im, ax=ax, shrink=0.45)
    row_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    column_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    ax.set_xticks(np.arange(len(row_labels)), column_labels, fontsize='9', rotation='vertical')
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontsize='9')

plt.tight_layout()
plt.show()

# Part 2 : Data of Medication ON
"""
-1)  Read the epoched data.
-2)  Concatenate the data (to convert it into continuous data).
-3)  Z-score the continuous data.
-4)  Epoch the z-scored continuous data.
-5)  Reshape the epoched and z-scored time series to (trials, times, channels).
"""
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']
continuous_on = {}
transpose_on = {}
z_continuous_on = {}
raw_on = {}
event_array = {}
epochs = {}
event_id = {'event_type': 1} 
tmin, tmax = 0, 4
sfreq = 512
original_ch_names = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'Fz', 'F4', 'C3', 'C4', 'Cz']
new_ch_names = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
ch_types = ['eeg'] * len(original_ch_names)  

 
sampling_infos = {}

for subject in subjects:
#Read the preprocessed and epoched data 
    info_original = mne.create_info(original_ch_names, sfreq, ch_types=ch_types, verbose=None)
    med_on_file_path = f"/home/schweyantoine/Bureau/Data/RESTING-DATA/preprocessed_data/dataClean_step1/dataClean_step1_{subject}_MED_ON.mat"
    epochs_on = mne.io.read_epochs_fieldtrip(med_on_file_path, info_original, data_name='dataClean_step1', trialinfo_column=6)

#Rearrange the channel order to facilitate interpretation later on with the transition matrices
    channel_mapping = [original_ch_names.index(ch) for ch in new_ch_names]
    rearranged_data = epochs_on.get_data()[:, channel_mapping, :]
    epochs[subject] = {'ON': rearranged_data}

# Concatenate and transpose the epoched data to obtain continuous data
    continuous_on[subject] = np.concatenate(epochs[subject]['ON'], axis=1)
    transpose_on[subject] = np.transpose(continuous_on[subject], (1, 0))
    
# Z-score the continuous data
    z_continuous_on[subject] = stats.zscore(transpose_on[subject], axis=0)
    z_continuous_on[subject] = np.transpose(z_continuous_on[subject], (1, 0))

# Epoch the z-scored continuous data
    info_new = mne.create_info(new_ch_names, sfreq, ch_types)
    raw_on[subject] = mne.io.RawArray(z_continuous_on[subject], info_new)

# Read the sampling info
    med_on = f"/home/schweyantoine/Bureau/Data/RESTING-DATA/preprocessed_data/dataClean_step1/dataClean_step1_{subject}_MED_ON.mat"
    mat_med_on = mat73.loadmat(med_on)
    sampling_infos[subject] = mat_med_on['dataClean_step1']['sampleinfo']

# Convert the sampling info to an event array
    event_array[subject] = np.column_stack(
        (sampling_infos[subject][:, 0].astype(int), np.zeros(len(sampling_infos[subject]), dtype=int),
         np.ones(len(sampling_infos[subject]), dtype=int)))

# Z-score the epoched data with MNE function
    epochs[subject] = mne.Epochs(raw_on[subject], events=event_array[subject], event_id=event_id, tmin=tmin,
                                  tmax=tmax, baseline=None, preload=True)

# Reshape the epoched and z-scored time series to (trials, times, channels). 
zscore_epoched_time_series_on = {}
for subject in subjects : 
    
    zscore_epoched_time_series_on [subject]=  np.transpose(epochs[subject].get_data(copy=True), (0, 2, 1))
    print (zscore_epoched_time_series_on[subject].shape)

# Compute avalanches for each subject in the medication ON condition
    # Compute transition matrices for each subject; shape is (n_channels, n_channels, n_avalanches)
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']

transition_matrices_dict_avalanches_on = {}

for subject in subjects:
      
    ntrials, ntimes, n_channels = zscore_epoched_time_series_on[subject].shape
    
    transition_matrices_on_list = []

    for trial in range(ntrials):
        avalanches, _, _, _, mat = find_avalanches(
            zscore_epoched_time_series_on[subject][trial, :, :], thresh=2, val_duration=2)
        
        # Each avalanche has a transition matrix of dimensions (nchannels, nchannels)
        for avalanche in avalanches:
            transition_matrices_on_list.append(mat)
              
    # Each subject has a 3D transition matrix (n channels, n channels, total number of avalanches)
    transition_matrices_on = np.stack(transition_matrices_on_list, axis=2)

   
    transition_matrices_dict_avalanches_on[subject] = transition_matrices_on
    print(transition_matrices_dict_avalanches_on[subject].shape)
    
# Plot the average transition matrix for each subject
vmin = {}
vmax = {}

num_subjects = len(subjects)
num_rows = (num_subjects // 3) + (num_subjects % 3 > 0)
num_cols = min(num_subjects, 3)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

for i, subject in enumerate(subjects):
    row, col = divmod(i, 3)  
    ax = axes[row, col] if num_subjects > 1 else axes  
    # Vmin and Vmax
    vmin[subject] = min(np.min(transition_matrices_dict_avalanches_off[subject].mean(axis=2)), np.min(transition_matrices_dict_avalanches_on[subject].mean(axis=2)))
    vmax[subject] = max(np.max(transition_matrices_dict_avalanches_off[subject].mean(axis=2)), np.max(transition_matrices_dict_avalanches_on[subject].mean(axis=2))) 
    # Plot the transition matrix: average across avalanches
    im = ax.imshow(transition_matrices_dict_avalanches_on[subject].mean(axis=2), cmap='viridis', interpolation='none', vmin=vmin[subject], vmax=vmax[subject])
    ax.set_title(f'Transition matrix for {subject} MED ON')
    fig.colorbar(im, ax=ax, shrink=0.45)
    row_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    column_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4', 'F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    ax.set_xticks(np.arange(len(row_labels)), column_labels, fontsize='9', rotation='vertical')
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontsize='9')
    
plt.tight_layout()
plt.show()

# Symmetrize and save the 3D matrices (n_channels, n_channels, total_number_avalanches) for each subject for both conditions (ON and OFF)
# Symmetrize and save the 3D matrices (nchannels, nchannels, n_avalanches)
# Function to symmetrize the 2D matrices
def symmetrize_matrix(matrix):
    return (matrix + matrix.T) / 2

# Function to symmetrize the 3D matrices
def symmetrize_3d_matrix(matrix_3d):
    return np.stack([symmetrize_matrix(matrix_3d[:, :, idx]) for idx in range(matrix_3d.shape[2])], axis=2)
symmetric_transition_matrices_dict_avalanches_on = {}
symmetric_transition_matrices_dict_avalanches_off = {}

for subject in subjects:

    symmetric_transition_matrices_dict_avalanches_on[subject] = symmetrize_3d_matrix(transition_matrices_dict_avalanches_on[subject])
    symmetric_transition_matrices_dict_avalanches_off[subject] = symmetrize_3d_matrix(transition_matrices_dict_avalanches_off[subject])

for subject in subjects:
    np.savez(f'{subject}_ATM_3D.npz',
             array_on=symmetric_transition_matrices_dict_avalanches_on[subject],
             array_off=symmetric_transition_matrices_dict_avalanches_off[subject])
# Check symmetry : By verifying the equality between the matrix and its transpose along the third dimension 
def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)
    for subject in subjects:

    for avalanche_idx in range(symmetric_transition_matrices_dict_avalanches_off[subject].shape[2]):
        if not is_symmetric(symmetric_transition_matrices_dict_avalanches_off[subject][:, :, avalanche_idx]):
            print(f"The symmetrized matrix for Subject {subject}, Avalanche {avalanche_idx} (off condition) is not symmetric.")
    for avalanche_idx in range(symmetric_transition_matrices_dict_avalanches_on[subject].shape[2]):
        if not is_symmetric(symmetric_transition_matrices_dict_avalanches_on[subject][:, :, avalanche_idx]):
            print(f"The symmetrized matrix for Subject {subject}, Avalanche {avalanche_idx} (on condition) is not symmetric.")

print("Symmetry check completed.")
