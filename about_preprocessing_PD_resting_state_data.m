%% ----------------------------- About -------------------------- %
%The preprocessing steps with the Fieldtrip Toolbox include filtering (high pass at 1.3 Hz), downsampling (from 2048 Hz to 512 Hz), and epoching (4 %seconds for each epoch).
%% ----------------------------- Authors -------------------------- %
% Hasnae AGOURAM
% Date: 13/06/2024
% License: BSD (3-clause)
%% ----------------------------- INITALISATION -------------------------- %

clear all;
close all;

%% ---------------------------------------------------------------------- %
addpath /home/user/Documents/MATLAB/fieldtrip-20210114
ft_defaults

% To adjust the position of the bowser figures; to be modified depending on your screen
fig_position_left = [0 750 900 700];
fig_position_right = [900 750 900 700];

%% ---------------------------------------------------------------------- %
% subject and condition lables 
sub_labels = { 's02','s04','s05','s06','s07','s08','s09','s10','s11','s14','s15','s16','s17','s18','s19'};
cond_labels = { 'MED_ON','MED_OFF'};
fig_cond_labels = { 'MED_ON','MED_OFF'};

%% -------------------- DIRECTORY TO SAVE MATLAB DATA ------------------- %
%  To store your matlab data
dir_saving = [ '/home/user/Bureau/Data/RESTING-DATA/preprocessed_data' ]
mkdir([ dir_saving '/dataContinuous_filtred' ])
mkdir( [dir_saving '/dataEpochedConc'] )

%% ------------------------ IMPORT Fieldtrip RAW EEG DATA ------------------------- %

for sub = 1 :length(sub_labels)
    
    for cond = 1 :length(cond_labels)
        
        dir_fieldtrip_raw_eeg = [ '/home/user/Bureau/Data/RESTING-DATA/preprocessed_data/fieldtrip_struct/'  sub_labels{sub} '/' cond_labels{cond} ]
        cd(dir_fieldtrip_raw_eeg);

        sub_labels{sub}
        cond_labels{cond}
        load ('resting_ft.mat')

%%------------------------------Preprocessing---------------------------------------%%       
         
         %---------------------Filtering-----------------------%
        cfg = [];
        cfg.hpfilter = 'yes'; % apply Highpass filter at 1.3 Hz
        cfg.hpfreq = 1.3;     % set filter frequency range
        cfg.dftfilter = 'yes';	% enable notch filtering to eliminate power line noise
        cfg.dftfreq   = [50 ];  % I also apply the notch filter at 100 Hz for some patients.
        resting_ft_filter = ft_preprocessing(cfg, resting_ft ); 

        %------------------Downsampling ------------------------%
        rs = 512; % downsampling from 2048 Hz to 512 Hz
        cfg = [];
        cfg.resamplefs = rs;
        resting_ft_filterResampled = ft_resampledata(cfg, resting_ft_filter);

        %------------------Epoching-----------------------%

        epoch_length = 4; % length of each epoch is 4 seconds
        samples_per_epoch = epoch_length * resting_ft_filterResampled.fsample;
        n_epochs = floor(length(resting_ft_filterResampled.time{1}) / samples_per_epoch);

        trl = zeros(n_epochs, 3);
        for i = 1:n_epochs
            trl(i, :) = [((i-1)*samples_per_epoch+1) (i*samples_per_epoch) 0];
        end

        cfg = [];
        cfg.trl = trl;
        resting_ft_EpochedConc = ft_redefinetrial(cfg, resting_ft_filterResampled);

        % --------------------Save the data-------------------- %
        
        cd([ dir_saving '/dataEpochedConc' ])
        filename = ['resting_ft_EpochedConc_' sub_labels{sub} '_'  cond_labels{cond} ]
        eval( ['save ', filename , ' resting_ft_EpochedConc -v7.3'] )
        
    end
end

%% ---------------------------------------------------------------------- % 
  % Raw Data browsing (before applying the notch filtre)
        cfg = [];
        cfg.channel = 'all';
        cfg.viewmode = 'vertical';
        cfg.ylim = [-20 20];
        cfg.fontsize = 0.015;
        ft_databrowser(cfg,  resting_ft);
        set(gcf,'position',fig_position_left)


 % Data browsing (after applying the notch filter and high pass filter)
        cfg = [];
        cfg.channel = 'all';
        cfg.viewmode = 'vertical';
        cfg.ylim = [-20 20];
        cfg.fontsize = 0.015;
        ft_databrowser(cfg,  resting_ft_filter);
        set(gcf,'position',fig_position_left)



 % Data browsing (after applying the filtering and downsampling)
        cfg = [];
        cfg.channel = 'all';
        cfg.viewmode = 'vertical';
        cfg.ylim = [-20 20];
        cfg.fontsize = 0.015;
        ft_databrowser(cfg,  resting_ft_filterResampled);
        set(gcf,'position',fig_position_left)
        
% Data browsing (after epoching ) 
        cfg = [];
        cfg.channel = 'all';
        cfg.viewmode = 'vertical';
        cfg.ylim = [-20 20];
%         cfg.fontsize = 0.015;
        ft_databrowser(cfg,  resting_ft_EpochedConc);
        set(gcf,'position',fig_position_left)

 %------------------------------------------------------------------------%

 
