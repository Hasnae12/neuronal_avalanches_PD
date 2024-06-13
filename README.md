# L-Dopa induced changes in aperiodic bursts dynamics relate to individual clinical improvement in Parkinson's disease

---

This repository includes the code and figures associated with the manuscript below:

Hasnae Agouram; Matteo Neri; Marianna Angiolelli; Damien Depannemaecker; Jyotika Bahuguna; Antoine Schwey; Jean Régis; Romain Carron; Alexandre Eusebio; Emmanuel Daucé; Pierpaolo Sorrentino (2024). L-Dopa induced changes in aperiodic bursts dynamics relate to individual clinical improvement in Parkinson's disease.

---

# Authors
Hasnae Agouram1,3*, Matteo Neri1, Marianna Angiolelli2,6, Damien Depannemaecker2, Jyotika Bahuguna5, Antoine Schwey1, Jean Régis8, Romain Carron2,7, Alexandre Eusebio1,4, Nicole Malfait, Emmanuel Daucé1,3¡  and Pierpaolo Sorrentino2*¡ 


1. Aix Marseille Univ, CNRS, INT, Institut de Neurosciences de la Timone, Marseille, France.
2. Aix-Marseille Univ, INSERM, INS, Institut de Neurosciences des Systèmes, Marseille, France
3. Ecole Centrale Méditerranée, Marseille, France. 	 	
4. APHM, Hôpitaux Universitaires de Marseille, Hôpital de la Timone Department of Neurology and Movement disorders, France.
5. Laboratoire de Neurosciences Cognitives et Adaptatives, Université de Strasbourg, Strasbourg, France.
6. Unit of Nonlinear Physics and Mathematical Models, Department of Engineering, Campus Bio-Medico University of Rome, 00128, Italy.
7. Medico-surgical Unit Epileptology, Functional and Stereotactic Neurosurgery, Timone University Hospital, Marseille, France.
8. Aix Marseille Univ, UMR INSERM 1106, Dept of Functional Neurosurgery, Marseille France.
¡ senior authors

---

# Abstract
Parkinson's disease (PD) is a neurodegenerative disease primarily characterized by severe motor symptoms that can be transiently relieved by medication (e.g. levodopa). These symptoms are mirrored by widespread alterations of neuronal activities across the whole brain, whose characteristics at the large scale level are still poorly understood. To address this issue, we measured the resting state activities of 11 PD patients utilizing different devices, i.e deep brain stimulation (DBS) contacts placed within the subthalamic nucleus area, and EEG electrodes placed above the motor areas. Data were recorded in each patient before drug administration (OFF-condition) and after drug administration (ON-condition). Neuronal avalanches, i.e. brief bursts of activity with widespread propagation, were detected and quantified on both types of contacts, and used to characterize differences in both conditions. Of particular interest, we noted a larger number of shorter and smaller avalanches in the OFF-condition, and a lesser number of wider and longer avalanches in the ON-condition. This difference turned out to be statistically significant at the group level. Then, we computed the avalanche transition matrices (ATM) to track the contact-wise patterns of avalanche spread. We compared the two conditions and observed a higher probability that an avalanche would spread within and between STN and motor cortex in the ON-state, with highly significant differences at the group level. Furthermore, we discovered that the increase in overall propagation of avalanches was correlated to clinical improvement after levodopa administration. Our results provide the first cross-modality assessment of aperiodic activities in PD patients, and the first account of the changes induced by levodopa on cross-regional aperiodic bursts at the individual level, and could open new avenues toward developing biomarkers of PD electrophysiological alterations.

# Analysis

1) The file **code1-Subject_level_analysis_from_time_series_to_avalanche_transition_matrices** : To compute the avalache transition matrices (ATMs) from the time series for each subject.
   
2) The file **code2_statistical analysis at the subject level** : To perform the permutation test at the subject level to obtain significant edges of the transition matrices for each subject, along with additional analyses for consistently significant edges across subjects.

3) The file **code3_avalanche analysis at the group level**: To compute the difference between the average ATMs ON-levodopa and OFF-levodopa across all patients.

4) The file **code4_stats_analysis_ATMs_averages_values**: To compare the ATM in the ON-levodopa and OFF-levodopa conditions for each patient (averages and values of the Cortex-Cortex edges, STN-Cortex edges, and STN-STN edges). We assess the significance of the differences using Wilcoxon signed-rank test.

5) The file **code5_robust_linear_regression_ATM_clinical_data**: To assess the correlation between the clinical improvement and the ATM averages ON/OFF ratio using robust linear regression, as well as the correlation of the clinical improvement with subgroups of edges (cortex_cortex,STN-cortex, STN-STN).

6) The file **code6_neuronal_avalanches_features_number_size_duration_inter-avalanche-interval**: To compare the number, size, duration, and inter-avalanche interval between ON and OFF conditions at both the subject and group levels.
   
7) The file **about_preprocessing_PD_resting_state_data**: To describe the preprocessing steps using the Fieldtrip Toolbox.

           1. **Filtering:** Apply a high-pass filter at 1.3 Hz. 
           2. **Downsampling:** Downsample from 2048 Hz to 512 Hz.
           3. **Epoching:** 4 seconds for each epoch.
           4. **Remove the bad epochs.**
