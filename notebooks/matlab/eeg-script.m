function [eeg_data, roi_activity] = simulate_eeg_from_network(fmri_neural_signals, connectivity_matrix)
% SIMULATE_EEG_FROM_NETWORK Generate simulated EEG data from fMRI neural signals
%
% Inputs:
%   fmri_neural_signals - Neural signals from fMRI (5 ROIs x 200 timepoints)
%   connectivity_matrix - Ground truth directed network connectivity matrix
%
% Outputs:
%   eeg_data - Simulated EEG data (channels x 6000 timepoints)
%   roi_activity - Latent ROI activity (3 ROIs x 6000 timepoints)

% Constants for EEG generation
n_rois_fmri = 5;           % Number of fMRI ROIs
n_rois_eeg = 3;            % Number of EEG ROIs (subset of fMRI ROIs)
n_times_fmri = 200;        % fMRI timepoints
n_times_eeg = 6000;        % EEG timepoints
n_channels = 64;           % Number of EEG channels

% Validate input dimensions
[rows_fmri, cols_fmri] = size(fmri_neural_signals);
if rows_fmri ~= n_rois_fmri || cols_fmri ~= n_times_fmri
    error('Expected fMRI neural signals to be %d ROIs x %d timepoints', n_rois_fmri, n_times_fmri);
end

% Select which fMRI ROIs to map to EEG ROIs (for example, first 3)
roi_mapping = 1:n_rois_eeg;  % Map first 3 fMRI ROIs to EEG ROIs

% Create time vectors for both modalities (normalized to [0,1])
t_fmri = linspace(0, 1, n_times_fmri);
t_eeg = linspace(0, 1, n_times_eeg);

% Interpolate fMRI neural signals to EEG time scale for selected ROIs
roi_activity = zeros(n_rois_eeg, n_times_eeg);
for roi = 1:n_rois_eeg
    fmri_roi = roi_mapping(roi);
    roi_activity(roi,:) = interp1(t_fmri, fmri_neural_signals(fmri_roi,:), t_eeg, 'pchip');
end

% Add event-related activity - extract events from fMRI data
% Find peaks in fMRI signal to identify events
events = [];
for roi = 1:n_rois_fmri
    [~, peaks] = findpeaks(fmri_neural_signals(roi,:), 'MinPeakHeight', mean(fmri_neural_signals(roi,:)) + std(fmri_neural_signals(roi,:)));
    events = unique([events, peaks]);
end

% Map fMRI event times to EEG timepoints
eeg_events = round(interp1(t_fmri, 1:n_times_fmri, t_eeg(events), 'nearest'));

% Generate EEG-specific events (higher temporal resolution)
n_extra_events = 50;  % Add more events only visible in EEG
extra_events = sort(randi([1, n_times_eeg], 1, n_extra_events));
all_eeg_events = sort(unique([eeg_events, extra_events]));

% Apply event-related modulation to ROI activity
event_duration = round(n_times_eeg / 100);  % Short event duration
for e = 1:length(all_eeg_events)
    event_start = all_eeg_events(e);
    event_end = min(event_start + event_duration, n_times_eeg);
    
    % Create small event-related perturbation
    for roi = 1:n_rois_eeg
        % Different amplitude for each ROI
        amplitude = 0.5 + 0.2*roi;
        
        % Apply event-related response with some jitter between ROIs
        for t = event_start:event_end
            time_offset = t - event_start;
            roi_activity(roi, t) = roi_activity(roi, t) + amplitude * exp(-(time_offset-3)^2/8);
        end
    end
end

% Generate lead field matrix: each ROI projects to a subset of channels
G = zeros(n_channels, n_rois_eeg);
ch_per_roi = ceil(n_channels / n_rois_eeg);

for roi = 1:n_rois_eeg
    % Select channels for this ROI with some spatial pattern
    start_ch = 1 + (roi-1) * ch_per_roi;
    end_ch = min(n_channels, roi * ch_per_roi);
    ch_range = start_ch:end_ch;
    
    % Create a realistic spatial pattern (strongest in the center, weaker at edges)
    for ch_idx = 1:length(ch_range)
        ch = ch_range(ch_idx);
        center = start_ch + floor(length(ch_range)/2);
        dist = abs(ch - center) / (length(ch_range)/2);
        G(ch, roi) = (1 - 0.7*dist^2);  % Gaussian-like spatial pattern
    end
    
    % Add some overlap between regions for realism
    if roi > 1
        overlap_start = max(1, start_ch - floor(ch_per_roi/4));
        for ch = overlap_start:start_ch-1
            dist = abs(ch - (start_ch-1)) / (ch_per_roi/4);
            G(ch, roi) = max(0, 0.4 * (1 - dist));
        end
    end
end

% Apply network effects based on connectivity matrix
% Extract the subnetwork connectivity for EEG ROIs
eeg_connectivity = connectivity_matrix(roi_mapping, roi_mapping);

% Simulate network dynamics
for t = 2:n_times_eeg
    % Apply network effects
    network_influence = eeg_connectivity * roi_activity(:, t-1);
    
    % Add to current activity (with dampening)
    roi_activity(:, t) = roi_activity(:, t) + 0.2 * network_influence;
end

% Add realistic EEG noise
% Pink noise (1/f) which is characteristic of EEG
noise_level = 0.2 * mean(std(roi_activity, [], 2));
noise = zeros(n_channels, n_times_eeg);

for ch = 1:n_channels
    % Generate white noise
    white_noise = randn(1, n_times_eeg);
    
    % Convert to pink noise in frequency domain
    noise_fft = fft(white_noise);
    freq = 1:length(noise_fft);
    scaling_factor = 1 ./ sqrt(freq + 1);
    scaling_factor(1) = 1;  % DC component
    
    % Apply 1/f scaling
    pink_noise_fft = noise_fft .* scaling_factor;
    pink_noise = real(ifft(pink_noise_fft));
    
    % Normalize and scale
    pink_noise = pink_noise / std(pink_noise) * noise_level;
    noise(ch, :) = pink_noise;
end

% Generate final EEG data
eeg_data = G * roi_activity + noise;

end