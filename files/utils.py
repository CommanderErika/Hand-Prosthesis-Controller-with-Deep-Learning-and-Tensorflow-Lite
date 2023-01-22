import os
import glob
import scipy.io

import numpy as np
import cv2 as cv
import librosa
from scipy import signal

import signal_preprocess


def create_image(sample, FREQ, window_size, image_path, hop_size, n_fft, win_length):
    
    """
        sample - matrix with values of one sample
        FREQ - sample's Frequency 
        window_size - Window's size.
    """
    
    # Window
    window = signal.get_window('boxcar', 1024)
    
    # Pre-process
    #sample = sample[500:]

      # Aplicação de um filtro de 40Hz - 160Hz
    #sos = scipy.signal.butter(2, (60, 200), btype='bandpass', fs=FREQ, output='sos')
    #filtered = scipy.signal.sosfilt(sos, sample)
    
    

    # Spectogram
    spectrogram = librosa.feature.melspectrogram(y=sample,
                                      sr=FREQ,
                                      fmin = 0,
                                      fmax = 500 // 2,
                                      hop_length = hop_size,
                                      win_length = win_length,
                                      n_fft = n_fft,
                                                )
    
    spectrogram = librosa.power_to_db(spectrogram)
    #.astype(np.float32)
    
    if not spectrogram.all():
        print("Teste")
        return False
    
    # Saving
    cv.imwrite(image_path, spectrogram)
    
    return True
    

def num_samples(cwd):
    
    total_num_samples = 0
    
    total_num_per_class = [0] * NUM_CLASS
    
    files = glob.glob(cwd+"*/*.mat")
    
    for file in files:
            
        # File path
        mat = scipy.io.loadmat(file)
            
        for i, clas in enumerate(CLASSES):
            
            # Get shape
            num_samples = mat[clas].shape[0]
                
            # Sum
            total_num_samples = total_num_samples + num_samples
            
            total_num_per_class[i] = total_num_per_class[i] + num_samples
                
    print(f"Total Number of samples: {total_num_samples}")
    
    for i, clas in enumerate(CLASSES):
        
        print(f"Number of samples from class {clas}: {total_num_per_class[i]}")

def get_shape(path_dir: str):
    
    """
        Get Dimensions.
    """
    
    # Files
    files = glob.glob(path_dir + "*/*.mat")
    
    # Classes
    #CLASSES = ['cyl_ch1', 'cyl_ch2', 'hook_ch1', 'hook_ch2', 'tip_ch1', 'tip_ch2', 'palm_ch1', 'palm_ch2', 'spher_ch1', 'spher_ch2', 'lat_ch1', 'lat_ch2']
    CLASSES = ['cyl_ch1', 'hook_ch1', 'tip_ch1', 'palm_ch1', 'spher_ch1', 'lat_ch1']
    #CLASSES = ['cyl_ch1', 'hook_ch1']
    
    # Num of samples
    num_samples = 0
    
    # Lenght
    #len_ = scipy.io.loadmat(files[0])[CLASSES[0]].shape[1]
    #print(len_)
    len_ = 3000

    # Counting
    for file in files:
        
        # Reading matlab file for each experiment
        mat = scipy.io.loadmat(file)
        
        # for every class
        for CLASS in CLASSES:
            
            num_samples += mat[CLASS].shape[0]
    
    return num_samples, len_



