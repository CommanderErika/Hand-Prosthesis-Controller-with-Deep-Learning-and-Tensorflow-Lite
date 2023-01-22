## functions to make signal preprocess
#
# 1. FFT
# 2. Bandpass filter + Retification
#  a. Butterworth BP filter, de 4 orden ou 6 orden
# 3. Full wave retification - Absolute value
# 4. Moving RMS Envelope
#  a. Mean Power 
#  b. Moving Average  with RMS (window = 50)
# 5. Normalisation
#  a. Maximum voluntary contration (MCV)
#  b. PDM
# 6. TKE

# SMG = (RMS envelope / MCV ) * 100

import numpy as np
from scipy import signal


def FFT(signal, sample_rate):
    
    """
        signal -
        sample_rate -
    """
    
    # Using numpy fft
    FTT = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1], d=1/sample_rate)
    
    return FTT, freq

def butter_bandpass_filer(signal_fft, sample_rate, sameple_rate, high, low, order):
    
    b, a = signal.butter(order, [low, high], fs = sample_rate, btype = 'bandpass', analog = True)
    
    return

def bandpass_filter():
    
    return 

def full_wave_retification(signal):
    
    """
        Apply absolute value in the signal.
        This is what happens on a full wave rectifier.
    """
    
    signal = [abs(value) for value in signal]
    
    return signal

def moving_rms(signal, window_size=50):
    
    """
        Moving RMS.
        signal -
        window_size - window size to apply RMS.
    """
    
    signal = np.power(signal, 2)
    window = np.ones(window_size)/float(window_size)
    
    return np.sqrt(np.convolve(signal, window, 'valid'))

        
        