#%%
"""
==============================================================

To compare the ATM in the ON-levodopa and OFF-levodopa conditions for each patient:

    Part1. Compare the ATM averages
    Part2. Compare the ATM values of the Cortex-Cortex edges, STN-Cortex edges, and STN-STN edges

    *Use Wilcoxon signed-rank test

===============================================================

"""
 
# Authors:
# Hasnae AGOURAM   
# The functions of the avalanche transition matrix are carried out by Pierpaolo Sorrentino
# The function to display the bar with significance levels is carrie out by Matteo Neri
# Date: 13/06/2024
# License: BSD (3-clause)

# Import libraries
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

# 1)- Read the 3D matrices :(n_channels, n_channels,number_of_avalanches) for each subject
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']

loaded_data = {}
ATM_3D = {}

for subject in subjects:
    file_path = f'{subject}_ATM_3D.npz'
    loaded_data[subject] = np.load(file_path)
    ATM_3D[subject] = {'ON': loaded_data[subject]['array_on'], 'OFF': loaded_data[subject]['array_off']}

# 2)- Average of transition matrices along the third dimension (number of avalanches) for each subject and condition
    # While averaging along the third dimension, we don't consider the zero elements.
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']

ATM_2D = {}

for subject in subjects:
    ATM_3D_ON = loaded_data[subject]['array_on']
    ATM_3D_OFF = loaded_data[subject]['array_off']
    
    ATM_2D[subject] = {
       'ON': np.sum (ATM_3D_ON, axis=2) / np.sum(ATM_3D_ON != 0, axis=2),
       'OFF': np.sum (ATM_3D_OFF, axis=2) / np.sum(ATM_3D_OFF != 0, axis=2)
    }

# 3)- Plot the 2D transition matrices for each subject

for subject, data in ATM_2D.items():
     
    ATM_2D_ON = data['ON']
    ATM_2D_OFF = data['OFF']

     
    vmin_subject = min(np.min(ATM_2D_ON), np.min(ATM_2D_OFF))
    vmax_subject = max(np.max(ATM_2D_ON), np.max(ATM_2D_OFF))

     
    plt.subplot(1, 2, 1)
    plt.imshow(ATM_2D_ON, cmap='viridis', interpolation='none', vmin=vmin_subject, vmax=vmax_subject)
    plt.colorbar(shrink=0.45)
    row_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4','F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    column_labels = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4','F3', 'C3', 'F4', 'C4', 'Fz', 'Cz']
    plt.xticks(np.arange(len(row_labels)), column_labels, fontsize='9', rotation='vertical')
    plt.yticks(np.arange(len(row_labels)), row_labels, fontsize='9')
    plt.title(f'Transition matrix for {subject} MED ON')

    plt.subplot(1, 2, 2)
    plt.imshow(ATM_2D_OFF, cmap='viridis', interpolation='none', vmin=vmin_subject, vmax=vmax_subject)
    plt.colorbar(shrink=0.45)
    plt.xticks(np.arange(len(row_labels)), column_labels, fontsize='9', rotation='vertical')
    plt.yticks(np.arange(len(row_labels)), row_labels, fontsize='9')
    plt.title(f'Transition matrix for {subject} MED OFF')

    plt.tight_layout()
    plt.show()

# Part1. Compare the ATM averages
ATM_average_ON = {}
ATM_average_OFF = {}

ATM_average_list_ON = []
ATM_average_list_OFF = []

for subject in subjects:
    # ON condition
    ATM_average_ON[subject] = np.mean(ATM_2D[subject]['ON'], axis=(0, 1))  
    ATM_average_list_ON.append(ATM_average_ON[subject])
    
    # OFF condition
    ATM_average_OFF[subject] = np.mean(ATM_2D[subject]['OFF'], axis=(0, 1))  
    ATM_average_list_OFF.append(ATM_average_OFF[subject])


# Function to display the bar with significance levels
def bars_diff(p, bottom, top,x1=1, x2=2, height=1):
    # Get info about y-axis
    yrange = top - bottom
   
    # What level is this bar among the bars above the plot?
    level = 1
    # Plot the bar
    bar_height = (yrange * 0.08 * level)*height + top
    bar_tips = bar_height - (yrange * 0.02)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    # Significance level

    if p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = ''
    text_height = bar_height + (yrange * 0.01)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')
     

# Compute the average over all edges in the 'ON' and 'OFF' conditions for each subject, and perform a statistical test for group-level analysis
serie1 =  ATM_average_list_ON
serie2 =  ATM_average_list_OFF
 
plt.figure(figsize=(3.3, 3.3))
plt.boxplot([serie1 ,serie2],patch_artist=False,labels=['ON', 'OFF'])
plt.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
plt.title('Average over all edges', fontsize = 15)
label1='ON'
label2='OFF'

plt.xticks([1, 2], [label1, label2], fontsize = 15)
#plt.xlabel('Condition', fontsize = 10)
plt.ylabel('ATM averages', fontsize = 15)

stats,p_values = sp.stats.wilcoxon(serie1 ,serie2)
print(p_values)
subject_labels = ['S02', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10', 'S15', 'S16', 'S17', 'S18']
plt.xticks([1, 2], [label1, label2], fontsize = 15)
plt.yticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3], [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], fontsize = 13)

for i in range(len(serie1)):
        point1=serie1[i]
        point2=serie2[i]

        plt.scatter(1,point1,color='c',alpha=.4)
        plt.scatter(2,point2,color='r',alpha=.4)
        #plt.text(1, point1, subject_labels[i], ha='right', va='center', color='c', fontsize=6)
        #plt.text(2, point2, subject_labels[i], ha='left', va='center', color='r', fontsize=6)
            
        if point1-point2 >0:
            plt.plot([1,2],[point1,point2],color='grey',alpha=0.3)
        else:
            plt.plot([1,2],[point1,point2],color='grey',alpha=0.3,linestyle='--')

to_min1=np.array([serie1, serie2])
bottom1, top1 = np.min(to_min1), np.max(to_min1)
bars_diff(p_values, bottom1, top1,height=3)
plt.ylim((0.04,0.3))
plt.tight_layout()
plt.savefig("ATM_averages_group_level.jpg", dpi=600)    
plt.show()

# Part2. Compare the ATM values of the Cortex-Cortex edges, STN-Cortex edges, and STN-STN edges
subjects = ['s02', 's04', 's05', 's06', 's07', 's08', 's10', 's15', 's16', 's17', 's18']
submatrix_indices = {}
average_matrix_on = {}
average_matrix_off = {}
for subject in subjects:
    submatrix_indices[subject] = [((0, 7), (0, 7)), ((0, 7), (8, 13)), ((8, 13), (0, 7)), ((8, 13), (8, 13))]
    average_matrix_off[subject] = np.zeros((2, 2))
    average_matrix_on[subject]  = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            submatrix = submatrix_indices[subject][i * 2 + j]
            rows1, rows2 = submatrix[0]
            cols1, cols2 = submatrix[1]
            submatrix_off = ATM_2D[subject]['OFF'][rows1:rows2 + 1, cols1:cols2 + 1]
            average_matrix_off[subject][i, j] = np.mean(submatrix_off)
            submatrix_on = ATM_2D[subject]['ON'][rows1:rows2 + 1, cols1:cols2 + 1]
            average_matrix_on[subject][i, j] = np.mean(submatrix_on)
    print( average_matrix_off[subject].shape,  average_matrix_on[subject].shape)

# Boxplots of comparison of the ATM values of the Cortex-Cortex edges, STN-Cortex edges, and STN-STN edges
for i in range(2):
    for j in range(2):
        ATM_values_list_ON = [average_matrix_on[subject][i, j] for subject in subjects]
        ATM_values_list_OFF = [average_matrix_off[subject][i, j] for subject in subjects]
        
        
        serie1 = ATM_values_list_ON

        serie2= ATM_values_list_OFF

        label1='ON'
        label2='OFF'
        plt.figure(figsize=(3.3, 3.3))

        box_plot_data=[serie1,serie2] 
        
        subject_labels = ['S02', 'S04', 'S05', 'S06', 'S07', 'S08', 'S10', 'S15', 'S16', 'S17', 'S18']
        plt.xticks([1, 2], [label1, label2], fontsize = 15)
        plt.yticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3], [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], fontsize = 13)

        for k in range(len(serie1)):
            point1=serie1[k]
            point2=serie2[k]
            plt.scatter(1, point1, color='c', alpha=0.5, s=15)  
            plt.scatter(2, point2, color='r', alpha=0.5, s=15)

            #plt.text(1, point1, subject_labels[k], ha='right', va='center', color='c')
            #plt.text(2, point2, subject_labels[k], ha='left', va='center', color='r')
    
            if point1-point2 >0:
                plt.plot([1,2],[point1,point2],color='grey',alpha=0.4)
            else:
                plt.plot([1,2],[point1,point2],color='grey',alpha=0.4,linestyle='--')

        plt.boxplot(box_plot_data,patch_artist=False,labels=[label1, label2])
        plt.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
        stats,pp = sp.stats.wilcoxon(ATM_values_list_ON, ATM_values_list_OFF)
        print(stats,pp)
        to_min1=np.array([serie1, serie2])
        bottom1, top1 = np.min(to_min1), np.max(to_min1)
        bars_diff(pp, bottom1, top1,height=3)
        
        title_list = ['STN-STN edges', 'STN-Cortex edges', 'STN-Cortex edges', 'Cortex-Cortex edges'] 
        plt.title(title_list[i * 2 + j], fontsize = 15)
        plt.ylim((0.04,0.3))
        plt.ylabel ('ATM values', fontsize = 15)
        plt.tight_layout()
        plt.savefig('EEG_STN_connections'+str(i)+str(j), dpi=600)
        plt.show()
        plt.close()  