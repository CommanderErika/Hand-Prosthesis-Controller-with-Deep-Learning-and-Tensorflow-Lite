# Creating database based on the dataset
import os
import glob
import shutil

import scipy.io
import numpy as np

import signal_preprocess
from utils import create_image, get_shape

def lstm_database(path_dir: str, sample_rate: int):
    
    """
        This function is to create a database by getting a path to the signals.
        path_dir: string.
        sample_rate: int.
    """
    
    # Get X dimensions
    shape = get_shape(path_dir)
    
    # Inputs
    X_1 = np.zeros(shape)
    print(X_1.shape)
    X_2 = np.zeros(shape)
    print(X_2.shape)
    
    # Target 
    y = np.array([])
    print(y.shape)
    
    # Classes
    CLASSES_1 = ['cyl_ch1','hook_ch1','tip_ch1','palm_ch1','spher_ch1','lat_ch1']
    CLASSES_2 = ['cyl_ch2','hook_ch2','tip_ch2','palm_ch2','spher_ch2','lat_ch2']
    
    # Map
    dict_ = {'cyl_ch1': 0,
             'cyl_ch2': 0,
             'hook_ch1': 1,
             'hook_ch2': 1,
             'tip_ch1': 2,
             'tip_ch2': 2,
             'palm_ch1': 3,
             'palm_ch2': 3,
             'spher_ch1': 4,
             'spher_ch2': 4,
             'lat_ch1': 5,
             'lat_ch2': 5       
    }
    
    # Getting path to every .mat file on path_emg
    files = glob.glob(path_dir + "*/*.mat")
    
    # Index
    idx = 0
    
    # Applying pre-process for every sample
    for file in files:
        
        # Reading matlab file for each experiment
        mat = scipy.io.loadmat(file)
        
        # for every class
        for CLASS_1, CLASS_2 in zip(CLASSES_1, CLASSES_2):
            
            # Number of samples per class
            num_samples = mat[CLASS_1].shape[0]
            
            # For every sample in the class
            for i in range(num_samples):
            
                # Sample
                sample_1 = mat[CLASS_1][i, :]
                sample_2 = mat[CLASS_2][i, :]

                # FFT
                sample_fft_1, freq_1 = signal_preprocess.FFT(sample_1, sample_rate=sample_rate)
                sample_fft_2, freq_2 = signal_preprocess.FFT(sample_2, sample_rate=sample_rate)

                # Full wave retification (Absolute value)
                sample_abs_1 = signal_preprocess.full_wave_retification(sample_fft_1)
                sample_abs_2 = signal_preprocess.full_wave_retification(sample_fft_2)

                # Moving RMS
                sample_rms_1 = signal_preprocess.moving_rms(sample_abs_1, window_size=50)
                sample_rms_2 = signal_preprocess.moving_rms(sample_abs_2, window_size=50)
                
                # X
                X_1[idx, 0:(len(sample_rms_1))] = sample_rms_1
                X_2[idx, 0:(len(sample_rms_2))] = sample_rms_2
                
                # target
                y = np.append(y, [int(dict_[CLASS_1])], axis=0)
                
                # Counte idx
                idx += 1
                
    X_1 = np.reshape(X_1, (idx, 1, 3000))
    X_2 = np.reshape(X_2, (idx, 1, 3000))
                
    return X_1, X_2, y


def conv1d_database(path_dir: str, sample_rate: int):

    """
        This function is to create a database by getting a path to the signals.
        path_dir: string.
        sample_rate: int.
    """
    
    # Get X dimensions
    shape = get_shape(path_dir)
    
    # Inputs
    X_1 = np.zeros(shape)
    print(X_1.shape)
    X_2 = np.zeros(shape)
    print(X_2.shape)
    
    # Target 
    y = np.array([])
    print(y.shape)
    
    # Classes
    CLASSES_1 = ['cyl_ch1','hook_ch1','tip_ch1','palm_ch1','spher_ch1','lat_ch1']
    CLASSES_2 = ['cyl_ch2','hook_ch2','tip_ch2','palm_ch2','spher_ch2','lat_ch2']
    
    # Map
    dict_ = {'cyl_ch1': 0,
             'cyl_ch2': 0,
             'hook_ch1': 1,
             'hook_ch2': 1,
             'tip_ch1': 2,
             'tip_ch2': 2,
             'palm_ch1': 3,
             'palm_ch2': 3,
             'spher_ch1': 4,
             'spher_ch2': 4,
             'lat_ch1': 5,
             'lat_ch2': 5       
    }
    
    # Getting path to every .mat file on path_emg
    files = glob.glob(path_dir + "*/*.mat")
    
    # Index
    idx = 0
    
    # Applying pre-process for every sample
    for file in files:
        
        # Reading matlab file for each experiment
        mat = scipy.io.loadmat(file)
        
        # for every class
        for CLASS_1, CLASS_2 in zip(CLASSES_1, CLASSES_2):
            
            # Number of samples per class
            num_samples = mat[CLASS_1].shape[0]
            
            # For every sample in the class
            for i in range(num_samples):
            
                # Sample
                sample_1 = mat[CLASS_1][i, :]
                sample_2 = mat[CLASS_2][i, :]

                # Full wave retification (Absolute value)
                sample_abs_1 = signal_preprocess.full_wave_retification(sample_1)
                sample_abs_2 = signal_preprocess.full_wave_retification(sample_2)

                # Moving RMS
                sample_rms_1 = signal_preprocess.moving_rms(sample_abs_1, window_size=50)
                sample_rms_2 = signal_preprocess.moving_rms(sample_abs_2, window_size=50)
                
                # X
                X_1[idx, 0:(len(sample_rms_1))] = sample_1
                X_2[idx, 0:(len(sample_rms_2))] = sample_2
                
                # target
                y = np.append(y, [int(dict_[CLASS_1])], axis=0)
                
                # Counte idx
                idx += 1
                
    X_1 = np.reshape(X_1, (idx, 3000))
    X_2 = np.reshape(X_2, (idx, 3000))
                
    return X_1, X_2, y

def conv2d_database_v2(path_dir: str, sample_rate: int, window_size: int, hop_size: int, n_fft: int, win_length: int):

    # Channel 1
    CLASSES_1 = ['cyl_ch1','hook_ch1',
                'tip_ch1',
               # 'palm_ch1',
                'spher_ch1','lat_ch1']
    # Channel 2
    CLASSES_2 = ['cyl_ch2','hook_ch2',
                'tip_ch2',
              #  'palm_ch2',
                'spher_ch2','lat_ch2']

    # X1, X2, y
    X1 = np.array([])
    X2 = np.array([])
    y = np.array([])

    # Getting path to every .mat file on path_emg
    files = glob.glob(path_dir + "*/*.mat")

    # Count to name each sample
    count = 0
   
    # Classes
    # Resolve this problem later
    CLASSES = ['cyl_ch1', 'cyl_ch2', 'hook_ch1', 'hook_ch2',
                'tip_ch1',
                'tip_ch2',
              #  'palm_ch1', 'palm_ch2',
            'spher_ch1', 'spher_ch2', 'lat_ch1', 'lat_ch2']


    # Directory for images
    if os.path.exists(path_dir + 'images'):
        print(f"Directory {path_dir + 'images'} already exist")
    else:
        os.mkdir(path_dir + 'images')
    
    path_dir = path_dir + 'images/'

    # Create two directory to separete for each channel
    # images/ch1 and images/ch2
    if os.path.exists(path_dir + 'ch1'):
        print(f"Directory {path_dir + 'ch1'} already exist")
    else:
        os.mkdir(path_dir + 'ch1')

    if os.path.exists(path_dir + 'ch2'):
        print(f"Directory {path_dir + 'ch2'} already exist")
    else:
        os.mkdir(path_dir + 'ch2')

    # Create directory for every class
    for CLASS in CLASSES:
        
        if CLASS[-1] == '1':
            if os.path.exists(path_dir + 'ch1/' + CLASS[:-4]):
                print(f"Directory {path_dir + 'ch1/' + CLASS[:-4]} already exist")
        
            else: 
                os.mkdir(path_dir + 'ch1/' + CLASS[:-4])
        elif CLASS[-1] == '2':
            if os.path.exists(path_dir + 'ch2/' + CLASS[:-4]):
                print(f"Directory {path_dir + 'ch2/' + CLASS[:-4]} already exist")
        
            else: 
                os.mkdir(path_dir + 'ch2/' + CLASS[:-4])

    # For every file
    for file in files:

        # Reading matlab file for each experiment
        mat = scipy.io.loadmat(file)

        # Fr every ch1 and ch2 in each class
        for ch1, ch2 in zip(CLASSES_1, CLASSES_2):

            # Number of samples per class
            num_samples = mat[CLASS].shape[0]

            for i in range(num_samples):

                # img for channel 1
                sample = mat[ch1][i, :] # sample for ch1
                #sample, freq_1 = np.asarray(signal_preprocess.FFT(sample, sample_rate=sample_rate), dtype=np.float32)
                #sample = np.asarray(signal_preprocess.full_wave_retification(sample))
                #sample = np.asarray(signal_preprocess.moving_rms(sample, window_size=20))
                image_path = image_path = path_dir + 'ch1/' + ch1[:-4] + "/" + str(count) + '.png'

                flag1 = create_image(sample=sample,
                             FREQ=sample_rate,
                             window_size=window_size,
                             image_path= image_path, # This CLASS is from CLASSES not CLASSES_
                             hop_size=hop_size,
                             n_fft = n_fft,
                             win_length = win_length
                            )

                # img for channel 2
                sample = mat[ch2][i, :] # sample for ch1
                #sample, freq_2 = np.asarray(signal_preprocess.FFT(sample, sample_rate=sample_rate), dtype=np.float32)
                #sample = np.asarray(signal_preprocess.full_wave_retification(sample))
                #sample = np.asarray(signal_preprocess.moving_rms(sample, window_size=20))
                image_path = image_path = path_dir + 'ch2/' + ch2[:-4] + "/" + str(count) + '.png'

                flag2 = create_image(sample=sample,
                             FREQ=sample_rate,
                             window_size=window_size,
                             image_path= image_path, # This CLASS is from CLASSES not CLASSES_
                             hop_size=hop_size,
                             n_fft = n_fft,
                             win_length = win_length
                            )
                if (not flag1) or (not flag2):
                    print("Alguma imagem com defeito, será descartada.")
                else:
                    # Counter
                    count += 1

def conv2d_database_v3(path_dir: str, sample_rate: int, window_size: int, hop_size: int, n_fft: int, win_length: int):

    # Channel 1
    CLASSES_1 = ['cyl_ch1','hook_ch1',
                #'tip_ch1',
                'palm_ch1','spher_ch1','lat_ch1']
    # Channel 2
    CLASSES_2 = ['cyl_ch2','hook_ch2',
                #'tip_ch2',
                'palm_ch2','spher_ch2','lat_ch2']

    # X1, X2, y
    X1 = np.array([])
    X2 = np.array([])
    y = np.array([])

    # Getting path to every .mat file on path_emg
    files = glob.glob(path_dir + "*/*.mat")

    # Count to name each sample
    count = 0
   
    # Classes
    # Resolve this problem later 
    CLASSES = ['cyl_ch1', 'cyl_ch2', 'hook_ch1', 'hook_ch2',
                'tip_ch1',
                'tip_ch2',
                #'palm_ch1',
                #'palm_ch2',
                'spher_ch1', 'spher_ch2', 'lat_ch1', 'lat_ch2']


    # Directory for images
    if os.path.exists(path_dir + 'images'):
        print(f"Directory {path_dir + 'images'} already exist")
    else:
        os.mkdir(path_dir + 'images')
    
    path_dir = path_dir + 'images/'

    # Create two directory to separete for each channel
    # images/ch1 and images/ch2
    if os.path.exists(path_dir + 'ch1'):
        print(f"Directory {path_dir + 'ch1'} already exist")
    else:
        os.mkdir(path_dir + 'ch1')

    if os.path.exists(path_dir + 'ch2'):
        print(f"Directory {path_dir + 'ch2'} already exist")
    else:
        os.mkdir(path_dir + 'ch2')

    # Create directory for every class
    for CLASS in CLASSES:
        
        if CLASS[-1] == '1':
            if os.path.exists(path_dir + 'ch1/' + CLASS[:-4]):
                print(f"Directory {path_dir + 'ch1/' + CLASS[:-4]} already exist")
        
            else: 
                os.mkdir(path_dir + 'ch1/' + CLASS[:-4])
        elif CLASS[-1] == '2':
            if os.path.exists(path_dir + 'ch2/' + CLASS[:-4]):
                print(f"Directory {path_dir + 'ch2/' + CLASS[:-4]} already exist")
        
            else: 
                os.mkdir(path_dir + 'ch2/' + CLASS[:-4])

    # For every file
    for file in files:

        # Reading matlab file for each experiment
        mat = scipy.io.loadmat(file)

        # Fr every ch1 and ch2 in each class
        for ch1, ch2 in zip(CLASSES_1, CLASSES_2):

            # Number of samples per class
            num_samples = mat[CLASS].shape[0]
            #print(f"Classe: {CLASS}, num de amostras: {num_samples}, nome do arquivo: {file}")
            #print(f"Classe: {ch1}, {ch2}")

            for i in range(num_samples):

                # samples
                # img for channel 1
                sample_1 = mat[ch1][i, :] # sample for ch1
                # img for channel 2
                sample_2 = mat[ch2][i, :] # sample for ch1
                
                if sample_1.shape[0] == 2500:
                    continue
                if sample_2.shape[0] == 2500:
                    continue
                
                print(sample_1.shape, sample_2.shape)
                
                list_1 = split_sample_in5(sample_1)
                list_2 = split_sample_in5(sample_2)
                
                print(list_1.shape, list_2.shape)
                
                for s1, s2 in zip(list_1, list_2):
                    #sample, freq_1 = np.asarray(signal_preprocess.FFT(sample, sample_rate=sample_rate), dtype=np.float32)
                    #sample = np.asarray(signal_preprocess.full_wave_retification(sample))
                    #sample = np.asarray(signal_preprocess.moving_rms(sample, window_size=20))
                    image_path = image_path = path_dir + 'ch1/' + ch1[:-4] + "/" + str(count) + '.png'
    
                    flag1 = create_image(sample=s1,
                                 FREQ=sample_rate,
                                 window_size=window_size,
                                 image_path= image_path, # This CLASS is from CLASSES not CLASSES_
                                 hop_size=hop_size,
                                 n_fft = n_fft,
                                 win_length = win_length
                                )
    
                    #sample, freq_2 = np.asarray(signal_preprocess.FFT(sample, sample_rate=sample_rate), dtype=np.float32)
                    #sample = np.asarray(signal_preprocess.full_wave_retification(sample))
                    #sample = np.asarray(signal_preprocess.moving_rms(sample, window_size=20))
                    image_path = image_path = path_dir + 'ch2/' + ch2[:-4] + "/" + str(count) + '.png'
    
                    flag2 = create_image(sample=s2,
                                 FREQ=sample_rate,
                                 window_size=window_size,
                                 image_path= image_path, # This CLASS is from CLASSES not CLASSES_
                                 hop_size=hop_size,
                                 n_fft = n_fft,
                                 win_length = win_length
                                )
                    if (not flag1) or (not flag2):
                        print("Alguma imagem com defeito, será descartada.")
                    else:
                        # Counter
                        count += 1
                        
    print(f"Quantidade de amostras: {count}")
                    
                    
def split_sample_in5(sample):
    
    # cut 1s from the sample
    if len(sample) == 3000:
        sample = sample[500:]
        
    list_sample = np.reshape(sample, (5, -1))
        
    return list_sample