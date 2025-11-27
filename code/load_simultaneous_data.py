import scipy.io as io
import numpy as np
import os
import mne

class SimultaneousData:
    """
    Represents and loads data for a single subject from the Simultaneous EEG&NIRS dataset.
    """
    def __init__(self, subject_id, task='nback', data_dir=r'd:\workspace\data\Simultaneous EEG&NIRS'):
        """
        Load and process one subject's data.

        Parameters:
        - subject_id: str, subject identifier (e.g., 'VP001')
        - task: str, task name ('dsr', 'nback', 'wg')
        - data_dir: str, base directory for the dataset
        """
        self.subject_id = subject_id
        self.task = task
        self.data_dir = data_dir
        
        # Load EEG data
        eeg_path = os.path.join(data_dir, f"{subject_id}-EEG")
        self.eeg_cnt = io.loadmat(os.path.join(eeg_path, f"cnt_{task}.mat"))
        self.eeg_mrk = io.loadmat(os.path.join(eeg_path, f"mrk_{task}.mat"))
        self.eeg_mnt = io.loadmat(os.path.join(eeg_path, f"mnt_{task}.mat"))
        
        # Load NIRS data
        nirs_path = os.path.join(data_dir, f"{subject_id}-NIRS")
        self.nirs_cnt = io.loadmat(os.path.join(nirs_path, f"cnt_{task}.mat"))
        self.nirs_mrk = io.loadmat(os.path.join(nirs_path, f"mrk_{task}.mat"))
        self.nirs_mnt = io.loadmat(os.path.join(nirs_path, f"mnt_{task}.mat"))
        
        # Extract EEG structured data
        self.eeg_data_struct = self.eeg_cnt[f'cnt_{task}'][0, 0]
        self.nirs_data_struct = self.nirs_cnt[f'cnt_{task}'][0, 0]
        
        # Extract EEG marker structured data
        self.eeg_mrk_struct = self.eeg_mrk[f'mrk_{task}'][0, 0]
        self.nirs_mrk_struct = self.nirs_mrk[f'mrk_{task}'][0, 0]
        
    def get_eeg_signal(self):
        """
        Get EEG signal data.
        
        Returns:
        - signal: np.ndarray, shape (n_channels, n_samples)
        - fs: float, sampling frequency in Hz
        - channel_labels: list of str, channel names
        """
        signal = self.eeg_data_struct['x']
        fs = float(self.eeg_data_struct['fs'][0, 0])
        clab = self.eeg_data_struct['clab'].flatten()
        channel_labels = [str(c[0]) if len(c) > 0 else f'Ch{i}' for i, c in enumerate(clab)]
        
        return signal, fs, channel_labels
    
    def get_nirs_signal(self):
        """
        Get NIRS signal data (HbO and HbR).
        
        Returns:
        - hbo: np.ndarray, shape (n_samples, n_channels), oxyhemoglobin
        - hbr: np.ndarray, shape (n_samples, n_channels), deoxyhemoglobin
        - fs: float, sampling frequency in Hz
        - channel_labels: list of str, channel names
        """
        # NIRS data has processed oxy/deoxy fields, each is a nested struct
        oxy_struct = self.nirs_data_struct['oxy'][0, 0]
        deoxy_struct = self.nirs_data_struct['deoxy'][0, 0]
        
        hbo = oxy_struct['x']  # oxyhemoglobin
        hbr = deoxy_struct['x']  # deoxyhemoglobin
        fs = float(oxy_struct['fs'][0, 0])
        clab = oxy_struct['clab'].flatten()
        channel_labels = [str(c[0]) if len(c) > 0 else f'Ch{i}' for i, c in enumerate(clab)]
        
        return hbo, hbr, fs, channel_labels
    
    def get_markers(self, modality='eeg'):
        """
        Get event markers.
        
        Parameters:
        - modality: str, 'eeg' or 'nirs'
        
        Returns:
        - times: np.ndarray, event times in ms
        - events: np.ndarray, event codes
        """
        if modality.lower() == 'eeg':
            mrk = self.eeg_mrk_struct
        else:
            mrk = self.nirs_mrk_struct
            
        times = mrk['time'].flatten() if 'time' in mrk.dtype.names else np.array([])
        events = mrk['event']['desc'].flatten() if 'event' in mrk.dtype.names else np.array([])
        
        return times, events
    
    def get_event_labels(self, modality='eeg'):
        """
        Get detailed event labels and class information.
        
        Parameters:
        - modality: str, 'eeg' or 'nirs'
        
        Returns:
        - times: np.ndarray, event times in ms (shape: n_events,)
        - y: np.ndarray, binary class labels (shape: n_classes, n_events)
        - class_names: list of str, class name for each row in y
        - event_codes: np.ndarray, raw event codes (shape: n_events,)
        """
        if modality.lower() == 'eeg':
            mrk = self.eeg_mrk_struct
        else:
            mrk = self.nirs_mrk_struct
        
        times = mrk['time'].flatten() if 'time' in mrk.dtype.names else np.array([])
        y = mrk['y'] if 'y' in mrk.dtype.names else np.array([])
        
        # Extract class names
        class_names = []
        if 'className' in mrk.dtype.names:
            for cn in mrk['className'].flatten():
                if len(cn) > 0:
                    class_names.append(str(cn[0]))
        
        # Extract event codes  
        event_codes = np.array([])
        if 'event' in mrk.dtype.names and 'desc' in mrk['event'].dtype.names:
            event_codes = mrk['event']['desc'][0, 0].flatten()
        
        return times, y, class_names, event_codes
    
    def create_epoch_labels(self, signal_length, fs, modality='eeg', target_classes=None, use_eeg_events=True):
        """
        Create sample-wise labels for the signal based on event markers.
        
        For n-back task:
        - target_classes can be e.g. ['0-back target', '2-back target', '3-back target']
          to create binary labels for target vs non-target trials.
        
        Parameters:
        - signal_length: int, number of samples in the signal
        - fs: float, sampling frequency of the output signal
        - modality: str, 'eeg' or 'nirs' (used for timing if use_eeg_events=False)
        - target_classes: list of str, class names to mark as positive (1)
        - use_eeg_events: bool, whether to use EEG events (more detailed) even for NIRS timing
        
        Returns:
        - labels: np.ndarray, shape (signal_length,), sample-wise labels
        - epoch_info: dict, containing epoch start/end indices and class info
        """
        # Always use EEG events since they have detailed trial info
        if use_eeg_events:
            times, y, class_names, event_codes = self.get_event_labels('eeg')
            # Get EEG sampling rate for time conversion
            eeg_fs = float(self.eeg_data_struct['fs'][0, 0])
        else:
            times, y, class_names, event_codes = self.get_event_labels(modality)
            eeg_fs = fs
        
        # Default: use 'target' classes (not 'non-target')
        if target_classes is None:
            target_classes = [cn for cn in class_names if 'target' in cn.lower() and 'non-target' not in cn.lower()]
        
        print(f"Target classes: {target_classes}")
        
        # Convert times from ms to samples at output fs
        event_samples = (times / 1000.0 * fs).astype(int)
        
        # Create sample-wise labels (0 = baseline, 1 = target event window)
        labels = np.zeros(signal_length, dtype=int)
        epoch_info = {'starts': [], 'ends': [], 'classes': [], 'class_names': class_names}
        
        # Mark event windows (e.g., 2 seconds after each event)
        window_samples = int(2.0 * fs)  # 2 second window
        
        target_count = 0
        for i, sample in enumerate(event_samples):
            if sample >= signal_length:
                continue
            
            end_sample = min(sample + window_samples, signal_length)
            y_row = y[:, i] if i < y.shape[1] else np.zeros(len(class_names))
            
            # Check if this event belongs to a target class
            is_target = False
            event_class = 'unknown'
            for j, cn in enumerate(class_names):
                if j < len(y_row) and y_row[j] == 1:
                    event_class = cn
                    if cn in target_classes:
                        is_target = True
                    break
            
            if is_target:
                labels[sample:end_sample] = 1
                target_count += 1
            
            epoch_info['starts'].append(sample)
            epoch_info['ends'].append(end_sample)
            epoch_info['classes'].append(event_class)
        
        print(f"Found {target_count} target epochs")
        return labels, epoch_info
    
    def get_cognitive_load_labels(self, signal_length, fs, modality='eeg', use_eeg_events=True):
        """
        Create labels based on cognitive load levels (0-back, 2-back, 3-back).
        
        Returns:
        - labels: np.ndarray, shape (signal_length,), cognitive load level (0, 2, or 3)
        - n_back_level: int for each sample
        """
        if use_eeg_events:
            times, y, class_names, _ = self.get_event_labels('eeg')
        else:
            times, y, class_names, _ = self.get_event_labels(modality)
        
        # Convert times from ms to samples
        event_samples = (times / 1000.0 * fs).astype(int)
        
        # Create sample-wise labels for cognitive load
        labels = np.zeros(signal_length, dtype=int)
        
        # Map class names to load levels
        load_map = {
            '0-back session': 0, '0-back target': 0,
            '2-back session': 2, '2-back target': 2, '2-back non-target': 2,
            '3-back session': 3, '3-back target': 3, '3-back non-target': 3,
        }
        
        window_samples = int(2.0 * fs)  # 2 second window
        
        for i, sample in enumerate(event_samples):
            if sample >= signal_length:
                continue
            
            end_sample = min(sample + window_samples, signal_length)
            y_row = y[:, i] if i < y.shape[1] else np.zeros(len(class_names))
            
            for j, cn in enumerate(class_names):
                if j < len(y_row) and y_row[j] == 1:
                    if cn in load_map:
                        labels[sample:end_sample] = load_map[cn]
                    break
        
        return labels 

if __name__ == "__main__":
    # Test loading
    subject = 'VP001'
    task = 'nback'
    
    print(f"Loading {subject} {task}...")
    loader = SimultaneousData(subject, task)
    
    # Get EEG signal
    eeg_signal, eeg_fs, eeg_labels = loader.get_eeg_signal()
    print(f"\nEEG Signal:")
    print(f"  Shape: {eeg_signal.shape}")
    print(f"  Sampling Rate: {eeg_fs} Hz")
    print(f"  Channels: {len(eeg_labels)}")
    print(f"  First 3 channel labels: {eeg_labels[:3]}")
    
    # Get NIRS signal
    hbo, hbr, nirs_fs, nirs_labels = loader.get_nirs_signal()
    print(f"\nNIRS Signal:")
    print(f"  HbO Shape: {hbo.shape}")
    print(f"  HbR Shape: {hbr.shape}")
    print(f"  Sampling Rate: {nirs_fs} Hz")
    print(f"  Channels: {len(nirs_labels)}")
    print(f"  First 3 channel labels: {nirs_labels[:3]}")
    
    # Get markers
    eeg_times, eeg_events = loader.get_markers('eeg')
    print(f"\nEEG Markers:")
    print(f"  Number of events: {len(eeg_times)}")
    if len(eeg_times) > 0:
        print(f"  First 3 event times (ms): {eeg_times[:3]}")
        print(f"  First 3 event codes: {eeg_events[:3]}")
