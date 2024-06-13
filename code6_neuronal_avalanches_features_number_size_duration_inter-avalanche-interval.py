#
"""
==============================================================
Steps:

To perform avalanches analysis on the epoched and preprocessed resting-state dataset of 11 Parkinsonian patients under two conditions (medication ON and OFF), I follow these steps:

- Read the epoched data.
- Concatenate the data (to convert it into continuous data).
- Z-score the continuous data.
- Epoch the z-scored continuous data.
- Binarize the z-scored epoched data (using a threshold).
- Compute the number, size, duration and inter-avalanche interval (IAI) of avalanches
- Do Kolmogorov-Smirnov (K-S) test on size, duration and IAI (subject and group levels)
                                         

    Remark: I concatenate the epochs before z-scoring because z-scoring the epoched data introduces noise (edge effect). Then, I re-epoch the data because our preprocessed data is epoched data (after rejecting noisy segments). Afterward, I follow the procedure to compute the number, size, duration and inter-avalanche interval of avalanches.

===============================================================

"""
 
# Authors:
# Hasnae AGOURAM   
# The functions of the avalanche transition matrix are carried out by Pierpaolo Sorrentino
# The functions to Compute the size, duration and Inter avalanche interval of avalanches are carried out by Matteo Neri
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

import seaborn as sns  
import pandas as pd
from scipy.stats import mannwhitneyu

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
#Read the epoched data of the OFF medication condition 
    info_original = mne.create_info(original_ch_names, sfreq, ch_types=ch_types, verbose=None)
    med_off_file_path = f"/home/schweyantoine/Bureau/Data/RESTING-DATA/preprocessed_data/dataClean_step1/dataClean_step1_{subject}_MED_OFF.mat"
    epochs_off = mne.io.read_epochs_fieldtrip(med_off_file_path, info_original, data_name='dataClean_step1', trialinfo_column=6)
    
#Rearrange the order of the channels, to facilitate the interpretation of the transition matrices
    channel_mapping = [original_ch_names.index(ch) for ch in new_ch_names]
    rearranged_data = epochs_off.get_data(copy=False)[:, channel_mapping, :]
    epochs[subject] = {'OFF': rearranged_data}

# Concatenate and transpose the epoched data, to have continuous data
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

# Epoched and z-scored time series
    epochs[subject] = mne.Epochs(raw_off[subject], events=event_array[subject], event_id=event_id, tmin=tmin,
                                  tmax=tmax, baseline=None, preload=True)

# Reshape the epoched and z-scored time series to (trials, times, channels).
zscore_epoched_time_series_off = {}
for subject in subjects : 
     
    zscore_epoched_time_series_off [subject]=  np.transpose(epochs[subject].get_data(copy=True), (0, 2, 1))
    print (zscore_epoched_time_series_off[subject].shape)

# Part 2 : Data of Medication ON
"""
-1)  Read the epoched data.
-2)  Concatenate the data (to convert it into continuous data).
-3)  Z-score the continuous data.
-4) Epoch the z-scored continuous data.
-5) Reshape the epoched and z-scored time series to (trials, times, channels).
""""
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
#Read the epoched data of the ON medication condition 
    info_original = mne.create_info(original_ch_names, sfreq, ch_types=ch_types, verbose=None)
    med_on_file_path = f"/home/schweyantoine/Bureau/Data/RESTING-DATA/preprocessed_data/dataClean_step1/dataClean_step1_{subject}_MED_ON.mat"
    epochs_on = mne.io.read_epochs_fieldtrip(med_on_file_path, info_original, data_name='dataClean_step1', trialinfo_column=6)

#Rearrange the order of the channels, to facilitate the interpretation of the transition matrices
    channel_mapping = [original_ch_names.index(ch) for ch in new_ch_names]
    rearranged_data = epochs_on.get_data(copy=False)[:, channel_mapping, :]
    epochs[subject] = {'ON': rearranged_data}

#  Concatenate and transpose the epoched data, to have continuous data
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

# Epoched and z-scored time series
    epochs[subject] = mne.Epochs(raw_on[subject], events=event_array[subject], event_id=event_id, tmin=tmin,
                                  tmax=tmax, baseline=None, preload=True)

# Reshape the epoched and z-scored time series to (trials, times, channels)
zscore_epoched_time_series_on = {}
for subject in subjects : 
     
    zscore_epoched_time_series_on [subject]=  np.transpose(epochs[subject].get_data(copy=True), (0, 2, 1))
    print (zscore_epoched_time_series_on[subject].shape)

# Part 3 : Compute the size, duration and Inter avalanche interval of avalanches : MED ON/OFF
"""
Functions:

Neuronal avalanches are characterized by their size, s, defined as the number of channels recruited during the avalanche; their duration, and the inter-avalanche interval (IAI), defined as the time interval between two consecutive avalanches

The function go_avalanches returns a dictionary. The key 'IAI' is the list of inter-avalanche intervals (IAI). The key 'dur' is the list of durations of avalanches. The key 'siz' is the list of sizes of avalanches.
"""
def consecutiveRanges(a):
    n=len(a)
    length = 1;list = []
    if (n == 0):
        return list 
    for i in range (1, n + 1):
        if (i == n or a[i] - a[i - 1] != 1):
            if (length > 0):
                if (a[i - length]!=0):
                    temp = (a[i - length]-1, a[i - 1])
                    list.append(temp)
            length = 1
        else:
            length += 1
    return list

def go_avalanches(data, binsize, thre=3., direc=0):
   
    #method can be simple or area. It depends on which criteria we want the algorithm to work.
    #the data is given time x regions
   
    if direc==1:
        Zb=np.where(stats.zscore(data)>thre,1,0)
    elif direc==-1:
        Zb=np.where(stats.zscore(data)<-thre,1,0)
    elif direc==0:
        #my data is already z_scored, I am just binarizing the data 
        Zb=np.where(np.abs(data)>thre,1,0)
    else:
        print('wrong direc')
       
    nregions=len(data[0])
   
    #here we are changing the length of the Zb in such a way that can be binarized with the binsize.
    Zb=Zb[:(int(len(Zb[:,0])/binsize)*binsize),:]

    a=np.sum(Zb)/(Zb.shape[0]*Zb.shape[1])
    Zbin=np.reshape(Zb,(-1, binsize, nregions))
    Zbin=np.where(np.sum(Zbin, axis=1)>0,1,0)
   
    #The number of regions actives at each time steps
    dfb_ampl=np.sum(Zbin,axis=1).astype(float)
    #the number of regions actives at each time step, no zeros, no times where no regions are activated.
    dfb_a=dfb_ampl[dfb_ampl!=0]
   
    #for the bratio the formula used here is the exp of the mean of the log of the ratio between the number of region actives at time t_i/t_(i-1).
    bratio=np.exp(np.mean(np.log(dfb_a[1:]/dfb_a[:-1])))
    #indices of no avalanches
    NoAval=np.where(dfb_ampl==0)[0]
   
    #here we plot the binarized matrix
    """plt.figure(figsize=(12,8))
    plt.imshow(Zbin.T[:,:1000], aspect='auto', interpolation='none')
    plt.colorbar()
    plt.show()
    plt.close()"""
   
    inter=np.arange(1,len(Zbin)+1); inter[NoAval]=0
    Avals_ranges=consecutiveRanges(inter)
    Avals_ranges=Avals_ranges[1:] #remove the first for avoiding boundary effects
   
    #plt.plot(inter[:3000])
    #plt.show()
    #plt.close()
   
    Naval=len(Avals_ranges)   #number of avalanches
   
    Avalanches={'dur':[],'siz':[],'IAI':[],'ranges':Avals_ranges[:-1],'Zbin': Zbin,'bratio':bratio, 'onespercentage': a} #duration and size
    for i in range(Naval-1): #It goes till the second last avalanche for avoiding bondaries effects
        xi=Avals_ranges[i][0];xf=Avals_ranges[i][1]; xone=Avals_ranges[i+1][0]
        Avalanches['dur'].append(xf-xi)
        Avalanches['IAI'].append(xone-xf)
        Avalanches['siz'].append(len(np.where(np.sum(Zbin[xi:xf],axis=0)>0)[0]))
       
    return Avalanches

# Medication OFF
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']
avalanches_dict_off = {}

for subject in subjects:
    avalanches_list_off = []   
    for trial in range(len(zscore_epoched_time_series_off[subject][:, 0, 0])):
        avalanches = go_avalanches(zscore_epoched_time_series_off[subject][trial, :, :], binsize=1, thre=2,direc=0)
        avalanches_list_off.append(avalanches)
    
    avalanches_dict_off[subject] = avalanches_list_off
# Medication ON
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']
avalanches_dict_on = {}

for subject in subjects:
    avalanches_list_on = []   
    for trial in range(len(zscore_epoched_time_series_on[subject][:, 0, 0])):
        avalanches = go_avalanches(zscore_epoched_time_series_on[subject][trial, :, :], binsize=1, thre=2,direc=0)
        avalanches_list_on.append(avalanches)
    
    avalanches_dict_on[subject] = avalanches_list_on
"""
1) Comparison of number of avalanches ON Vs OFF
Compare the distribution of number of avalanches across trials for each subject:

Comparison of the distributions of the number of avalanches across segments for each patient in the ON-state and OFF-state, along with Mann-Whitney U test results, mean values, and standard deviations.
"""
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']
avalanche_number_on_list = {}
avalanche_number_off_list = {}
min_bin = {}
max_bin = {}
bins = {}
U_stat_number = {}
p_value_utest_number = {}

num_subjects = len(subjects)
num_cols = 2  
num_rows = (num_subjects + num_cols - 1) // num_cols 

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3*num_rows))

for i, subject in enumerate(subjects):
    row_idx = i // num_cols
    col_idx = i % num_cols
    
    avalanche_number_on_list[subject] = np.array([len(x['IAI']) for x in avalanches_dict_on[subject]])
    avalanche_number_off_list[subject] = np.array([len(x['IAI']) for x in avalanches_dict_off[subject]])
 
    # Mann-Whitney U test 
    U_stat_number[subject], p_value_utest_number[subject] = mannwhitneyu(avalanche_number_on_list[subject],
                                                                         avalanche_number_off_list[subject],
                                                                         alternative='two-sided')

    serie1 = avalanche_number_on_list[subject]
    serie2 = avalanche_number_off_list[subject]
    ax = axes[row_idx, col_idx] if num_subjects > 1 else axes
    ax.boxplot ([serie1,serie2], patch_artist =  False, labels= ['ON','OFF'])
    label1='ON'
    label2='OFF'

    min_length = min(len(serie1), len(serie2))
    for k in range(min_length):
        point1=serie1[k]
        point2=serie2[k]

        ax.scatter(1,point1,color='c',alpha=.5)
        ax.scatter(2,point2,color='r',alpha=.5)
            
        if point1-point2 >0:
            ax.plot([1,2],[point1,point2],color='grey',alpha=0.4)
        else:
            ax.plot([1,2],[point1,point2],color='grey',alpha=0.4,linestyle='--')
            
    if p_value_utest_number[subject] < 0.001:
        ax.text(1.5, max(max(serie1), max(serie2)),'***', color='red', fontsize=12, va='center', ha='center')
    elif p_value_utest_number[subject] < 0.01:
         ax.text(1.5, max(max(serie1), max(serie2)),'**', color='red', fontsize=12, va='center', ha='center')
    elif p_value_utest_number[subject] < 0.05:
        ax.text(1.5, max(max(serie1), max(serie2)),'*', color='red', fontsize=12, va='center', ha='center')
    elif p_value_utest_number[subject] > 0.05:
         ax.text(1.5, max(max(serie1), max(serie2)),'ns', color='red', fontsize=12, va='center', ha='center') 
 
    ax.grid()
    
    patients = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11']
    patient_id = patients[i]  
    ax.set_title(f"Number of Avalanches Across Segments for {patient_id }", fontsize =  13)
    ax.set_xlabel('Conditions', fontsize=12)
    ax.set_ylabel('Number of Avalanches', fontsize=12)
    ax.set_xticklabels([label1, label2], fontsize=15)
    #ax.set_ylim(0, 140)

plt.tight_layout(pad=0.5)
fig.savefig('Box plots Number of Avalanches_subject_level', dpi=600)
plt.show()

# Store the resulting p-values and significance levels
avalanches_ON = {}
avalanches_OFF = {}
U_stat_number= {}
p_value_utest_number= {}
list_p_values_number = []
list_symbole_number = []

for subject in subjects:
     
    num_avalanches_on = avalanche_number_on_list[subject]
    num_avalanches_off = avalanche_number_off_list[subject]

    U_stat_number[subject],  p_value_utest_number[subject] = mannwhitneyu(num_avalanches_on, num_avalanches_off, alternative='two-sided')
    
    list_p_values_number.append(p_value_utest_number[subject])

for p_value in list_p_values_number:
    if p_value < 0.001:
        list_symbole_number.append('***')
    elif p_value < 0.01:
        list_symbole_number.append('**')
    elif p_value < 0.05:
        list_symbole_number.append('*')
    else:
        list_symbole_number.append('ns')

print("List of p-values:", list_p_values_number)
print("List of symbols:", list_symbole_number)

# Compute the total number of avalanches for each subject

avalanche_number_on_list  =  {}
avalanche_number_off_list =  {}

avalanches_ON  =  {}
avalanches_OFF =  {}

total_number_avalanches_ON =  {}
total_number_avalanches_OFF =  {}

list_total_number_avalanches_ON =  []
list_total_number_avalanches_OFF =  []

for subject in subjects:

    avalanche_number_on_list[subject] = [len(x['IAI']) for x in avalanches_dict_on[subject]]   
    avalanche_number_off_list[subject] = [len(x['IAI']) for x in avalanches_dict_off[subject]]
    
    avalanches_ON[subject]  = np.sum(avalanche_number_on_list[subject], axis=0)
    avalanches_OFF[subject] = np.sum(avalanche_number_off_list[subject], axis=0)


    total_number_avalanches_ON[subject] =  avalanches_ON[subject] 
    total_number_avalanches_OFF[subject] = avalanches_OFF[subject] 
    list_total_number_avalanches_ON.append(total_number_avalanches_ON[subject])
    list_total_number_avalanches_OFF.append(total_number_avalanches_OFF[subject])
 
print(list_total_number_avalanches_ON, list_total_number_avalanches_OFF)

# Compute the total Number of trials ON Vs. OFF for each subject
number_trials_on  =  []
number_trials_off =  []
for subject in subjects: 

    number_trials_on.append(zscore_epoched_time_series_on[subject].shape [0])
    number_trials_off.append(zscore_epoched_time_series_off[subject].shape [0])
print(number_trials_on,number_trials_off)

# 2) Comparison of size of avalanches ON Vs OFF

    # Plot distribution of comparison of size of avalanches between ON and OFF conditions at the subject level
avalanches_ON = {}
avalanches_OFF = {}
min_bin = {}
max_bin = {}
bins_size = {}
sample_size_ON = {}
sample_size_OFF = {}
U_stat_size = {}
p_value_utest_size = {}
df_size = {}


num_subjects = len(subjects)
num_cols = 2  
num_rows = (num_subjects + num_cols - 1) // num_cols 


fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3*num_rows))

patients = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11"]

for i, subject in enumerate(subjects):
    row_idx = i // num_cols
    col_idx = i % num_cols
     
    
    avalanches_ON[subject] = [x['siz'] for x in avalanches_dict_on[subject]]
    avalanches_OFF[subject] = [x['siz'] for x in avalanches_dict_off[subject]]
    
   
    sample_size_ON[subject] = np.concatenate(avalanches_ON[subject])
    sample_size_OFF[subject] = np.concatenate(avalanches_OFF[subject])

    U_stat_size[subject], p_value_utest_size[subject] = stats.ks_2samp(sample_size_ON[subject], sample_size_OFF[subject])


    min_bin[subject] = min(min(sample_size_ON[subject]), min(sample_size_OFF[subject]))
    max_bin[subject] = max(max(sample_size_ON[subject]), max(sample_size_OFF[subject]))
    bins_size[subject] = np.linspace(min_bin[subject], max_bin[subject], 15)

    counts_off_size, _ = np.histogram(sample_size_OFF[subject], bins=bins_size[subject], density=True)
    cdf_off_size = np.cumsum(counts_off_size * np.diff(bins_size[subject]))

    counts_on_size, _ = np.histogram(sample_size_ON[subject], bins=bins_size[subject], density=True)
    cdf_on_size = np.cumsum(counts_on_size * np.diff(bins_size[subject]))

    ax = axes[row_idx, col_idx] if num_subjects > 1 else axes
    ax.plot(bins_size[subject][1:], cdf_on_size, marker='o', linestyle='-', color='blue', alpha=0.8, label='ON')
    ax.plot(bins_size[subject][1:], cdf_off_size, marker='o', linestyle='-', color='orange', alpha=0.8, label='OFF')

    ax.text(0.5, 0.1, f'Mean size ON: {np.mean(sample_size_ON[subject]):.2f}\nMean size OFF: {np.mean(sample_size_OFF[subject]):.2f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, 0.3, f'STD ON: {np.std(sample_size_ON[subject]):.2f}\nSTD OFF: {np.std(sample_size_OFF[subject]):.2f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, 0.5, f'P_value: {p_value_utest_size[subject]:.3f}', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
 
    patient_id = patients[i]   
    ax.set_title(f"Size of avalanches across segments for {patient_id}" ,fontsize = 15)
    ax.set_xlabel("Avalanche Size (Number)",fontsize = 13)
    ax.set_ylabel("Cumulative Probability",fontsize = 13)
    ax.grid(True)
    #ax.legend()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.12, 0.98), fancybox=True, shadow=True)

plt.tight_layout(pad=0.5)
fig.savefig('CDF_features_of_Avalanches_subject_level_size.jpg', dpi = 600)
plt.show()


# Comparison of the average of size of avalanches ON Vs OFF for each subject
# Mean of the size of avalanches 
avalanches_ON  =  {}
avalanches_OFF =  {}
total_size_avalanches_ON=  {}
total_size_avalanches_OFF=  {}

list_total_size_avalanches_ON =  []
list_total_size_avalanches_OFF=  []

for subject in subjects: 

    avalanches_ON[subject] = [x['siz'] for x in avalanches_dict_on[subject]]
    avalanches_OFF[subject] = [x['siz'] for x in avalanches_dict_off[subject]]

    total_size_avalanches_ON[subject] = np.mean(np.concatenate(avalanches_ON[subject], axis=0))
    total_size_avalanches_OFF[subject] = np.mean(np.concatenate(avalanches_OFF[subject], axis=0))

    list_total_size_avalanches_ON.append(total_size_avalanches_ON[subject])
    list_total_size_avalanches_OFF.append(total_size_avalanches_OFF[subject])

print(list_total_size_avalanches_ON,list_total_size_avalanches_OFF)

# Store the resulting p-values and significance levels
avalanches_ON = {}
avalanches_OFF = {}
sample_size_ON = {}
sample_size_OFF = {}
U_stat_size = {}
p_value_utest_size = {}
df_size = {}
list_p_values_size=  []
list_symbole_size = []

for subject in subjects:
    avalanches_ON[subject] = [x['siz'] for x in avalanches_dict_on[subject]]
    avalanches_OFF[subject] = [x['siz'] for x in avalanches_dict_off[subject]]
    
    sample_size_ON[subject] = np.concatenate(avalanches_ON[subject], axis=0)  
    sample_size_OFF[subject] = np.concatenate(avalanches_OFF[subject], axis=0)

    # KS test (two-sample)
    U_stat_size[subject], p_value_utest_size[subject] = stats.ks_2samp(sample_size_ON[subject], sample_size_OFF[subject])
    list_p_values_size.append(p_value_utest_size[subject])
    
print(list_p_values_size)

for i in range (len(list_p_values_size)):   
    if list_p_values_size[i] < 0.001:
        list_symbole_size.append('***')
    elif list_p_values_size[i] < 0.01:
         
        list_symbole_size.append('**')             
    
    elif list_p_values_size[i] < 0.05:
          list_symbole_size.append('*')
                  
    else:
         
        list_symbole_size.append( 'ns' )      

print(list_symbole_size)

# 3) Compare the duration of avalanches ON Vs. OFF
avalanches_ON = {}
avalanches_OFF = {}
min_bin = {}
max_bin = {}
bins_duration = {}
sample_duration_ON = {}
sample_duration_OFF = {}
U_stat_duration = {}
p_value_utest_duration = {}
df_duration = {}

num_subjects = len(subjects)
num_cols = 2  
num_rows = (num_subjects + num_cols - 1) // num_cols 
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10,3*num_rows))

patients = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11"]

for i, subject in enumerate(subjects):
    row_idx = i // num_cols
    col_idx = i % num_cols
    
    avalanches_ON[subject] = [x['dur'] for x in avalanches_dict_on[subject]]
    avalanches_OFF[subject] = [x['dur'] for x in avalanches_dict_off[subject]]
    
    sample_duration_ON[subject] = np.concatenate(avalanches_ON[subject])
    sample_duration_OFF[subject] = np.concatenate(avalanches_OFF[subject])
    
    U_stat_duration[subject], p_value_utest_duration[subject] = stats.ks_2samp(sample_duration_ON[subject], sample_duration_OFF[subject])

    min_bin[subject] = min(min(sample_duration_ON[subject]), min(sample_duration_OFF[subject]))
    max_bin[subject] = max(max(sample_duration_ON[subject]), max(sample_duration_OFF[subject]))
    bins_duration[subject] = np.linspace(min_bin[subject], max_bin[subject], 15)

    counts_off_duration, _ = np.histogram(sample_duration_OFF[subject], bins=bins_duration[subject], density=True)
    cdf_off_duration = np.cumsum(counts_off_duration * np.diff(bins_duration[subject]))

    counts_on_duration, _ = np.histogram(sample_duration_ON[subject], bins=bins_duration[subject], density=True)
    cdf_on_duration = np.cumsum(counts_on_duration * np.diff(bins_duration[subject]))

    ax = axes[row_idx, col_idx] if num_subjects > 1 else axes
    ax.plot(bins_duration[subject][1:], cdf_on_size, marker='o', linestyle='-', color='blue', alpha=0.8, label='ON')
    ax.plot(bins_duration[subject][1:], cdf_off_size, marker='o', linestyle='-', color='orange', alpha=0.8, label='OFF')

    ax.text(0.5, 0.1, f'Mean duration ON: {np.mean(sample_duration_ON[subject]):.2f}\nMean duration OFF: {np.mean(sample_duration_OFF[subject]):.2f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, 0.3, f'STD ON: {np.std(sample_duration_ON[subject]):.2f}\nSTD OFF: {np.std(sample_duration_OFF[subject]):.2f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, 0.5, f'P_value: {p_value_utest_duration[subject]:.3f}', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    patient_id = patients[i]  
    ax.set_title(f"Duration of avalanches across segments for {patient_id}", fontsize=13)
    ax.set_xlabel("Duration (miliseconds)", fontsize=13)
    ax.set_ylabel("Cumulative Probability", fontsize=13)
    ax.grid(True)
    #ax.legend()


handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.12, 0.98), fancybox=True, shadow=True)

plt.tight_layout(pad=1)
plt.tight_layout()
fig.savefig('CDF_features_of_Avalanches_subject_level_duration.jpg', dpi=600)
plt.show()

### Store the resulting p-values and significance levels
avalanche_durations_on_list = {}
avalanche_durations_off_list= {}

sample_duration_ON = {}
sample_duration_OFF= {}
U_stat_duration= {}
p_value_utest_duration= {}
list_p_values_duration=  []
list_symbole_duration = []

for subject in subjects:
    avalanche_durations_on_list[subject] = [x['dur'] for x in avalanches_dict_on[subject]]
    avalanche_durations_off_list[subject] = [x['dur'] for x in avalanches_dict_off[subject]]
    sample_duration_ON[subject] = np.concatenate(avalanche_durations_on_list[subject], axis=0)
    sample_duration_OFF[subject] = np.concatenate(avalanche_durations_off_list[subject], axis=0)

    # K-S test (non-parametric)
    U_stat_duration[subject], p_value_utest_duration[subject] = stats.ks_2samp(sample_duration_ON[subject] , sample_duration_OFF[subject])
    
    list_p_values_duration.append(p_value_utest_duration[subject])
    
print(list_p_values_duration)

for i in range (len(list_p_values_duration)):   
    
    if list_p_values_duration[i] < 0.001:
        list_symbole_duration.append('***')
    
    elif list_p_values_duration[i] < 0.01:
         
        list_symbole_duration.append('**')             
    
    elif list_p_values_duration[i] < 0.05:
        
        list_symbole_duration.append('*')
                  
    else:
         
        list_symbole_duration.append( 'ns' )      

print(list_symbole_duration)


# Comparison of average of duration of avalanches for each subject

# Mean of duration of avalanches 
avalanche_durations_on_list  =  {}
avalanche_durations_off_list =  {}

avalanches_ON  =  {}
avalanches_OFF =  {}

total_duration_avalanches_ON =  {}
total_duration_avalanches_OFF =  {}

list_total_duration_avalanches_ON =  []
list_total_duration_avalanches_OFF =  []

for subject in subjects:

    avalanche_durations_on_list[subject] = [x['dur'] for x in avalanches_dict_on[subject]]
    avalanche_durations_off_list[subject] = [x['dur'] for x in avalanches_dict_off[subject]]
    
    avalanches_ON[subject]  = np.concatenate(avalanche_durations_on_list[subject], axis=0)
    avalanches_OFF[subject] = np.concatenate(avalanche_durations_off_list[subject], axis=0)


    total_duration_avalanches_ON[subject] = np.mean(avalanches_ON[subject], axis=0)
    total_duration_avalanches_OFF[subject] = np.mean(avalanches_OFF[subject], axis=0)
    list_total_duration_avalanches_ON.append(total_duration_avalanches_ON[subject])
    list_total_duration_avalanches_OFF.append(total_duration_avalanches_OFF[subject])

print(list_total_duration_avalanches_ON,list_total_duration_avalanches_OFF)

# 4) Comparison of INTER-AVALANCHE INTERVAL (IAI)

# Plot distribution of comparison of Inter-avalanche interval (IAI) of avalanches between ON and OFF conditions at the subject level
avalanches_ON = {}
avalanches_OFF = {}
min_bin = {}
max_bin = {}
bins_iai = {}
sample_iai_ON = {}
sample_iai_OFF = {}
U_stat_iai = {}
p_value_utest_iai = {}
df_iai = {}

num_subjects = len(subjects)
num_cols = 2  
num_rows = (num_subjects + num_cols - 1) // num_cols 
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3*num_rows))

patients = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11"]

for i, subject in enumerate(subjects):
    row_idx = i // num_cols
    col_idx = i % num_cols
    
    avalanches_ON[subject] = [x['IAI'] for x in avalanches_dict_on[subject]]
    avalanches_OFF[subject] = [x['IAI'] for x in avalanches_dict_off[subject]]
    
    sample_iai_ON[subject] = np.concatenate(avalanches_ON[subject])
    sample_iai_OFF[subject] = np.concatenate(avalanches_OFF[subject])
    
    U_stat_iai[subject], p_value_utest_iai[subject] = stats.ks_2samp(sample_iai_ON[subject], sample_iai_OFF[subject])

    min_bin[subject] = min(min(sample_iai_ON[subject]), min(sample_iai_OFF[subject]))
    max_bin[subject] = max(max(sample_iai_ON[subject]), max(sample_iai_OFF[subject]))
    bins_iai[subject] = np.linspace(min_bin[subject], max_bin[subject], 15)

    counts_off_iai, _ = np.histogram(sample_iai_OFF[subject], bins=bins_iai[subject], density=True)
    cdf_off_iai = np.cumsum(counts_off_iai * np.diff(bins_iai[subject]))

    counts_on_iai, _ = np.histogram(sample_iai_ON[subject], bins=bins_iai[subject], density=True)
    cdf_on_iai = np.cumsum(counts_on_iai * np.diff(bins_iai[subject]))

    ax = axes[row_idx, col_idx] if num_subjects > 1 else axes
    ax.plot(bins_iai[subject][1:], cdf_on_iai, marker='o', linestyle='-', color='blue', alpha=0.8, label='ON')
    ax.plot(bins_iai[subject][1:], cdf_off_iai, marker='o', linestyle='-', color='orange', alpha=0.8, label='OFF')
    
    ax.text(0.5, 0.1, f'Mean IAI ON: {np.mean(sample_iai_ON[subject]):.2f}\nMean IAI OFF: {np.mean(sample_iai_OFF[subject]):.2f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, 0.3, f'STD ON: {np.std(sample_iai_ON[subject]):.2f}\nSTD OFF: {np.std(sample_iai_OFF[subject]):.2f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(0.5, 0.5, f'P_value: {p_value_utest_iai[subject]:.4f}', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    patient_id = patients[i]  
    ax.set_title(f"IAI of avalanches across segments for {patient_id}", fontsize = 15)
    ax.set_xlabel("Inter-avalanche interval (miliseconds)", fontsize = 13)
    ax.set_ylabel("Cumulative Probability", fontsize = 13)
    ax.grid(True)
    #ax.legend(loc = (0.5, 0.6))

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.12, 0.98), fancybox=True, shadow=True)

plt.tight_layout(pad=0.5)
fig.savefig('CDF_features_of_Avalanches_subject_level_IAI.jpg', dpi=600)
plt.show()

# Comparison of average of Inter-avalanche interval between ON and OFF conditions for each subject
# Average Inter avalanche interval 
avalanche_iai_on_list  =  {}
avalanche_iai_off_list =  {}

avalanches_ON  =  {}
avalanches_OFF =  {}

total_iai_avalanches_ON =  {}
total_iai_avalanches_OFF =  {}

list_total_iai_avalanches_ON =  []
list_total_iai_avalanches_OFF =  []

for subject in subjects:

    avalanche_iai_on_list[subject] = [x['IAI'] for x in avalanches_dict_on[subject]]
    avalanche_iai_off_list[subject] = [x['IAI'] for x in avalanches_dict_off[subject]]
    
    avalanches_ON[subject]  = np.concatenate(avalanche_iai_on_list[subject], axis=0)
    avalanches_OFF[subject] = np.concatenate(avalanche_iai_off_list[subject], axis=0)


    total_iai_avalanches_ON[subject] = np.mean(avalanches_ON[subject], axis=0)
    total_iai_avalanches_OFF[subject] = np.mean(avalanches_OFF[subject], axis=0)
    list_total_iai_avalanches_ON.append(total_iai_avalanches_ON[subject])
    list_total_iai_avalanches_OFF.append(total_iai_avalanches_OFF[subject])

print(list_total_iai_avalanches_ON, list_total_iai_avalanches_OFF)

# Store the resulting p-values and significance levels
avalanche_iai_on_list = {}
avalanche_iai_off_list= {}

sample_iai_ON = {}
sample_iai_OFF= {}
U_stat_iai= {}
p_value_utest_iai= {}
list_p_values_iai=  []
list_symbole_iai = []
for subject in subjects:
    avalanche_iai_on_list[subject] = [x['IAI'] for x in avalanches_dict_on[subject]]
    avalanche_iai_off_list[subject] = [x['IAI'] for x in avalanches_dict_off[subject]]
    sample_iai_ON[subject] = np.concatenate(avalanche_iai_on_list[subject], axis=0)
    sample_iai_OFF[subject] = np.concatenate(avalanche_iai_off_list[subject], axis=0)

    # K-S test (non-parametric)
    U_stat_iai[subject], p_value_utest_iai[subject] = stats.ks_2samp(sample_iai_ON[subject] , sample_iai_OFF[subject])
    list_p_values_iai.append(p_value_utest_iai[subject])

print(list_p_values_iai)
for i in range (len(list_p_values_iai)):   
    if list_p_values_iai[i] < 0.001:
        list_symbole_iai.append('***')
    elif list_p_values_iai[i] < 0.01:
         
        list_symbole_iai.append('**')             
    
    elif list_p_values_iai[i] < 0.05:
        list_symbole_iai.append('*')
                  
    else:
         
        list_symbole_iai.append( 'ns' )      

print(list_symbole_iai)
"""
Subject level analysis of avalanches features

- Comparison between the number of avalanches per segment in ON-levodopa versus OFF-levodopa states for each patient. We assess the significance of the difference in the distributions of the number of avalanches across segments ON versus OFF using the Mann-Whitney test.  

- Comparison of the means of duration, size, and inter-avalanche interval,  for each patient. We assess the significance of the differences using the Kolmogorov-Smirnov (K-S) test between the distributions of ON-levodopa versus OFF-levodopa states for each patient. 

- The significance level are indicated as *** : p< 0.001, ** : p<0.01, *: p<0.05, ns = non-significant.
"""
bar_width = 0.4
index = np.arange(len(subjects))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

patients =  ["P01", "P02","P03","P04","P05","P06","P07","P08","P09","P10","P11"]

# Number of avalanches per trial ON Vs OFF
ax1.bar(index, np.divide(list_total_number_avalanches_ON, number_trials_on), bar_width,color='blue', alpha=0.8, label='ON')

_, caplines_number_avalanches_ON, _ = ax1.errorbar(index, np.divide(list_total_number_avalanches_ON, number_trials_on),
                                    yerr=np.std(np.divide(list_total_number_avalanches_ON, number_trials_on)),
                                    capsize=5, ecolor='silver', lolims =  True, uplims = False,ls='None') 

caplines_number_avalanches_ON[0].set_marker('_')

ax1.bar(index + bar_width, np.divide(list_total_number_avalanches_OFF, number_trials_off), bar_width,
          color='orange', alpha=0.8, label='OFF')

_, caplines_number_avalanches_OFF, _ = ax1.errorbar(index + bar_width, np.divide(list_total_number_avalanches_OFF, number_trials_off),
        yerr=np.std(np.divide(list_total_number_avalanches_OFF, number_trials_off)),
        capsize=5, ecolor='silver', lolims =  True, uplims = False, ls='None') 
caplines_number_avalanches_OFF[0].set_marker('_')

for i in range(len(subjects)):
    ax1.text(index[i] + bar_width / 2, max(np.divide(list_total_number_avalanches_ON, number_trials_on)[i], np.divide(list_total_number_avalanches_OFF, number_trials_off)[i])+16, list_symbole_number[i],
            ma = 'center',ha='center', va='bottom', color= 'black',fontsize = 12)

ax1.set_ylabel('# Avalanches per Segment', fontsize=15)
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(patients, rotation=60, fontsize = 12)
ax1.legend(loc='upper right', fontsize=10)
ax1.tick_params(axis='y', labelsize=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Mean duration ON Vs OFF 
ax2.bar(index, list_total_duration_avalanches_ON, bar_width,color='blue', label='ON')

_, caplines_duration_avalanches_ON, _ =  ax2.errorbar(index, list_total_duration_avalanches_ON, yerr=np.std(list_total_duration_avalanches_ON),
                                                      capsize=5, ecolor='silver', lolims =  True, uplims = False,ls='None')
caplines_duration_avalanches_ON[0].set_marker('_')
ax2.bar(index + bar_width, list_total_duration_avalanches_OFF, bar_width, color='orange', alpha=0.8, label='OFF')

_, caplines_duration_avalanches_OFF, _ =  ax2.errorbar(index + bar_width, list_total_duration_avalanches_OFF, 
        yerr=np.std(list_total_duration_avalanches_OFF),capsize=5, ecolor='silver', lolims =  True, uplims = False,ls='None') 

caplines_duration_avalanches_OFF[0].set_marker('_')

for i in range(len(subjects)):
    ax2.text(index[i] + bar_width / 2, max(list_total_duration_avalanches_ON[i], list_total_duration_avalanches_OFF[i]) + 3.8,
             list_symbole_duration[i], ma='center', ha='center', va='bottom', color='black', fontsize=12)

ax2.set_ylabel('<Duration>', fontsize=15)
ax2.set_xticks(index + bar_width / 2)
ax2.set_xticklabels(patients, rotation=60, fontsize = 12)
ax2.set_ylim((0, 29))
ax2.tick_params(axis='y', labelsize=12)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Mean size between ON Vs OFF
ax3.bar(index, list_total_size_avalanches_ON, bar_width, color='blue', alpha=0.8, label='ON')

_, caplines_size_avalanches_ON, _ = ax3.errorbar(index, list_total_size_avalanches_ON, yerr=np.std(list_total_size_avalanches_ON), 
                                                 capsize=5, ecolor='silver', lolims =  True, uplims = False,ls='None') 
caplines_size_avalanches_ON[0].set_marker('_')
ax3.bar(index + bar_width, list_total_size_avalanches_OFF, bar_width,color='orange', alpha=0.8, label='OFF')

_, caplines_size_avalanches_OFF, _ = ax3.errorbar(index + bar_width, list_total_size_avalanches_OFF, yerr=np.std(list_total_size_avalanches_OFF), 
                                                  capsize=5, ecolor='silver', lolims =  True, uplims = False,ls='None') 
caplines_size_avalanches_OFF[0].set_marker('_')
for i in range(len(subjects)):
    ax3.text(index[i] + bar_width / 2, max(list_total_size_avalanches_ON[i], list_total_size_avalanches_OFF[i]) + 0.55,
             list_symbole_size[i], ma='center', ha='center', va='bottom', color='black', fontsize=12)

ax3.set_xlabel('Patients', fontsize=15)
ax3.set_ylabel('<Size>', fontsize=15)
ax3.set_xticks(index + bar_width / 2)
ax3.set_xticklabels(patients, rotation=60, fontsize = 12)
ax3.set_ylim((0, 6.5))
ax3.tick_params(axis='y', labelsize=12)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Mean IAI ON Vs OFF
ax4.bar(index, list_total_iai_avalanches_ON, bar_width, color='blue', alpha=0.8, label='ON')

_, caplines_iai_avalanches_ON, _ = ax4.errorbar(index, list_total_iai_avalanches_ON, 
        yerr=np.std(list_total_iai_avalanches_ON),capsize=5, ecolor='silver', lolims =  True, uplims = False,ls='None') 

caplines_iai_avalanches_ON[0].set_marker('_')

ax4.bar(index + bar_width, list_total_iai_avalanches_OFF, bar_width, color='orange', alpha=0.8, label='OFF')

_, caplines_iai_avalanches_OFF, _ = ax4.errorbar(index + bar_width, list_total_iai_avalanches_OFF, 
        yerr=np.std(list_total_iai_avalanches_OFF),capsize=5, ecolor='silver', lolims =  True, uplims = False,ls='None') 

caplines_iai_avalanches_OFF[0].set_marker('_')

for i in range(len(subjects)):
    ax4.text(index[i] + bar_width / 2, max(list_total_iai_avalanches_ON[i], list_total_iai_avalanches_OFF[i]) + 21,
             list_symbole_iai[i], ma='center', ha='center', va='bottom', color='black', fontsize=12)

ax4.set_xlabel('Patients', fontsize=15)
ax4.set_ylabel('<IAI>', fontsize=15)
ax4.set_xticks(index + bar_width / 2)
ax4.set_xticklabels(patients, rotation=60, fontsize = 12)
ax4.set_ylim((0, 126))
ax4.tick_params(axis='y', labelsize=12)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

fig.tight_layout(pad=3)
plt.savefig("Avalanches_features_subject_level.jpg", dpi=600)
plt.show()

# Group level analysis of avalanches features: Number, duration, size, IAI of avalanches
"""
- Comparison of the distribution of number of avalanches per segment in ON-levodopa versus OFF-levodopa states at the group level. We assess the significance of the difference using the paired t-test. 

- Comparison of the cumulative distribution function at the group level for duration, size, and inter-avalanche interval. We assess the significance of the difference using the Kolmogorov-Smirnov (K-S) test. 

"""
# For number of avalanches
# Compute the total number of avalanches for each subject

avalanche_number_on_list  =  {}
avalanche_number_off_list =  {}

avalanches_ON  =  {}
avalanches_OFF =  {}

total_number_avalanches_ON =  {}
total_number_avalanches_OFF =  {}

list_total_number_avalanches_ON =  []
list_total_number_avalanches_OFF =  []

for subject in subjects:

    avalanche_number_on_list[subject] = [len(x['IAI']) for x in avalanches_dict_on[subject]]   
    avalanche_number_off_list[subject] = [len(x['IAI']) for x in avalanches_dict_off[subject]]
    
    avalanches_ON[subject]  = np.sum(avalanche_number_on_list[subject], axis=0)
    avalanches_OFF[subject] = np.sum(avalanche_number_off_list[subject], axis=0)


    total_number_avalanches_ON[subject] =  avalanches_ON[subject] 
    total_number_avalanches_OFF[subject] = avalanches_OFF[subject] 
    list_total_number_avalanches_ON.append(total_number_avalanches_ON[subject])
    list_total_number_avalanches_OFF.append(total_number_avalanches_OFF[subject])
 
print(list_total_number_avalanches_ON, list_total_number_avalanches_OFF)

# Compute the total Number of trials ON Vs. OFF for each subject
number_trials_on  =  []
number_trials_off =  []
for subject in subjects: 

    number_trials_on.append(zscore_epoched_time_series_on[subject].shape [0])
    number_trials_off.append(zscore_epoched_time_series_off[subject].shape [0])
print(number_trials_on,number_trials_off)

#Stat test on Total number of avalanches divided by number of trials for each condition and subject (because we don’t have the same exact number of trials for ON and OFF in each subject )
number_trials_on = np.array(number_trials_on)
number_trials_off = np.array(number_trials_off)

stats_number, p_value_number = stats.ttest_rel(np.divide(list_total_number_avalanches_ON, number_trials_on), np.divide(list_total_number_avalanches_OFF, number_trials_off))

print(p_value_number)

# For size of avalanches
avalanches_ON  =  {}
avalanches_OFF =  {}
total_size_avalanches_ON=  {}
total_size_avalanches_OFF=  {}

list_total_size_avalanches_ON =  []
list_total_size_avalanches_OFF=  []

for subject in subjects: 

    avalanches_ON[subject] = [x['siz'] for x in avalanches_dict_on[subject]]
    avalanches_OFF[subject] = [x['siz'] for x in avalanches_dict_off[subject]]

    total_size_avalanches_ON[subject]  = np.concatenate(avalanches_ON[subject], axis=0)
    total_size_avalanches_OFF[subject] = np.concatenate(avalanches_OFF[subject], axis=0)

    
    list_total_size_avalanches_ON.extend (total_size_avalanches_ON[subject])
    list_total_size_avalanches_OFF.extend(total_size_avalanches_OFF[subject])

print(len(list_total_size_avalanches_ON),len(list_total_size_avalanches_OFF))

# KS test (two-sample) at the group level 
U_stat_size, p_value_utest_size= stats.ks_2samp(list_total_size_avalanches_ON, list_total_size_avalanches_OFF)
print(U_stat_size,p_value_utest_size)

# For duration of avalanches
avalanches_ON  =  {}
avalanches_OFF =  {}
total_duration_avalanches_ON=  {}
total_duration_avalanches_OFF=  {}

list_total_duration_avalanches_ON =  []
list_total_duration_avalanches_OFF=  []

for subject in subjects: 

    avalanches_ON[subject] = [x['dur'] for x in avalanches_dict_on[subject]]
    avalanches_OFF[subject] = [x['dur'] for x in avalanches_dict_off[subject]]

    total_duration_avalanches_ON[subject]  = np.concatenate(avalanches_ON[subject], axis=0)
    total_duration_avalanches_OFF[subject] = np.concatenate(avalanches_OFF[subject], axis=0)

    
    list_total_duration_avalanches_ON.extend (total_duration_avalanches_ON[subject])
    list_total_duration_avalanches_OFF.extend(total_duration_avalanches_OFF[subject])

print(len(list_total_duration_avalanches_ON),len(list_total_duration_avalanches_OFF))

# KS test (two-sample) at the group level 
U_stat_duration, p_value_utest_duration= stats.ks_2samp(list_total_duration_avalanches_ON, list_total_duration_avalanches_OFF)
print(U_stat_duration,p_value_utest_duration)

# For inter-avalanche interval (IAI) of avalanches
avalanches_ON  =  {}
avalanches_OFF =  {}
total_iai_avalanches_ON=  {}
total_iai_avalanches_OFF=  {}

list_total_iai_avalanches_ON =  []
list_total_iai_avalanches_OFF=  []

for subject in subjects: 

    avalanches_ON[subject] = [x['IAI'] for x in avalanches_dict_on[subject]]
    avalanches_OFF[subject] = [x['IAI'] for x in avalanches_dict_off[subject]]

    total_iai_avalanches_ON[subject]  = np.concatenate(avalanches_ON[subject], axis=0)
    total_iai_avalanches_OFF[subject] = np.concatenate(avalanches_OFF[subject], axis=0)

    
    list_total_iai_avalanches_ON.extend (total_iai_avalanches_ON[subject])
    list_total_iai_avalanches_OFF.extend(total_iai_avalanches_OFF[subject])

print(len(list_total_iai_avalanches_ON),len(list_total_iai_avalanches_OFF))

# KS test (two-sample) at the group level 
U_stat_iai, p_value_utest_iai= stats.ks_2samp(list_total_iai_avalanches_ON, list_total_iai_avalanches_OFF)
print(U_stat_iai,p_value_utest_iai)

# Compute cumulative distribution function (CDF) for OFF and ON condition at the group level

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

#Number of avalanches 
serie1 = np.divide(list_total_number_avalanches_ON, number_trials_on)
serie2 = np.divide(list_total_number_avalanches_OFF, number_trials_off)

ax1.boxplot ([serie1,serie2], patch_artist =  False, labels= ['ON','OFF'])
label2='OFF'
ax1.set_xticks([1, 2], [label1, label2], fontsize = 15)
#ax1.set_yticks(fontsize = 10)
ax1.set_xlabel('Conditions', fontsize=15)
ax1.set_ylabel('# Avalanches per segment', fontsize=15)

for i in range(len(serie1)):
        point1=serie1[i]
        point2=serie2[i]

        ax1.scatter(1,point1,color='c',alpha=.5)
        ax1.scatter(2,point2,color='r',alpha=.5)
            
        if point1-point2 >0:
            ax1.plot([1,2],[point1,point2],color='grey',alpha=0.4)
        else:
            ax1.plot([1,2],[point1,point2],color='grey',alpha=0.4,linestyle='--')
            
if p_value_number < 0.001:
    ax1.text(1.5, max(max(serie1), max(serie2)),'***', color='red', fontsize=12, va='center', ha='center')
elif p_value_number < 0.01:
     ax1.text(1.5, max(max(serie1), max(serie2)),'**', color='red', fontsize=12, va='center', ha='center')
elif p_value_number < 0.05:
    ax1.text(1.5, max(max(serie1), max(serie2)),'*', color='red', fontsize=12, va='center', ha='center')
elif p_value_number > 0.05:
     ax1.text(1.5, max(max(serie1), max(serie2)),'ns', color='red', fontsize=12, va='center', ha='center') 

ax1.grid()
 

#  Duration of Avalanches
min_bin_duration = min(np.min(list_total_duration_avalanches_ON), np.min(list_total_duration_avalanches_OFF))
max_bin_duration = max(np.max(list_total_duration_avalanches_ON), np.max(list_total_duration_avalanches_OFF))
bins_duration = np.linspace(min_bin_duration, max_bin_duration, 30)

counts_off_duration,bin_edges_off_duration = np.histogram(list_total_duration_avalanches_OFF, bins=bins_duration, density=True)
cdf_off_duration = np.cumsum(counts_off_duration / np.sum(counts_off_duration)) 

counts_on_duration, bin_edges_on_duration = np.histogram(list_total_duration_avalanches_ON, bins=bins_duration, density=True)
cdf_on_duration = np.cumsum(counts_on_duration / np.sum(counts_on_duration)) 

ax2.plot(bin_edges_off_duration[1:], cdf_off_duration, marker='o', linestyle='-', color='orange', alpha=0.8, label='OFF')
ax2.plot(bin_edges_on_duration[1:], cdf_on_duration, marker='o', linestyle='-', color='blue', alpha=0.8,label='ON')  
ax2.text(0.75, 0.75, f'P_value: {p_value_utest_duration:.4f}', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.5))
ax2.set_xlabel('Duration of Avalanches (millisecond)', fontsize=15)
ax2.set_ylabel('Cumulative Probability', fontsize=15)

# Size of Avalanches
min_bin_size = min(np.min(list_total_size_avalanches_ON), np.min(list_total_size_avalanches_OFF))
max_bin_size = max(np.max(list_total_size_avalanches_ON), np.max(list_total_size_avalanches_OFF))
bins_size = np.linspace(min_bin_size, max_bin_size, 15)

counts_off_size, bin_edges_off_size = np.histogram(list_total_size_avalanches_OFF, bins=bins_size, density=True)
cdf_off_size = np.cumsum(counts_off_size / np.sum(counts_off_size))   

counts_on_size, bin_edges_on_size = np.histogram(list_total_size_avalanches_ON, bins=bins_size, density=True)
cdf_on_size = np.cumsum(counts_on_size / np.sum(counts_on_size)) 

ax3.plot(bin_edges_off_size[1:], cdf_off_size, marker='o', linestyle='-', color = 'orange', alpha=0.8, label='OFF')
ax3.plot(bin_edges_on_size[1:], cdf_on_size, marker='o', linestyle='-', color='blue', alpha=0.8,label='ON') 
ax3.text(0.75, 0.75, f'P_value: {p_value_utest_size:.4f}', transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.5))

ax3.set_xlabel('Size of Avalanches (number)', fontsize=15)
ax3.set_ylabel('Cumulative Probability', fontsize=15)
ax3.legend()

#  Inter-Avalanche Interval (IAI)
min_bin_iai = min(np.min(list_total_iai_avalanches_ON), np.min(list_total_iai_avalanches_OFF))
max_bin_iai = max(np.max(list_total_iai_avalanches_ON), np.max(list_total_iai_avalanches_OFF))
bins_iai = np.linspace(min_bin_iai, max_bin_iai, 30)

counts_off_iai, bin_edges_off_iai = np.histogram(list_total_iai_avalanches_OFF, bins=bins_iai, density=True)
cdf_off_iai = np.cumsum(counts_off_iai / np.sum(counts_off_iai)) 
counts_on_iai, bin_edges_on_iai = np.histogram(list_total_iai_avalanches_ON, bins=bins_iai, density=True)
cdf_on_iai = np.cumsum(counts_on_iai / np.sum(counts_on_iai))    

ax4.plot(bin_edges_off_iai[1:], cdf_off_iai, marker='o', linestyle='-',color = 'orange', alpha=0.8, label='OFF')
ax4.plot(bin_edges_on_iai[1:], cdf_on_iai, marker='o', linestyle='-', color='blue', alpha=0.8,  label='ON')  
ax4.text(0.75, 0.75, f'P_value: {p_value_utest_iai:.4f}', transform=ax4.transAxes, bbox=dict(facecolor='white', alpha=0.5))
ax4.set_xlabel('Inter-Avalanche Interval (millisecond)', fontsize=15)
ax4.set_ylabel('Cumulative Probability', fontsize=15)

plt.tight_layout(pad=1, w_pad=1, h_pad=1)
fig.savefig('CDF_features_of_Avalanches_Group_Level.jpg', dpi=600)
plt.show()



