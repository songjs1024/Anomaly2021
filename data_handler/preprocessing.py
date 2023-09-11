import numpy as np
from scipy.signal import butter
from scipy.signal import lfilter
import librosa

class preprocessing(object):

    def __init__(self,data,lowcut,highcut,fs,n_fft,hop_length,win_length,order=5):
        self.data = data
        self.lowcut = lowcut 
        self.highcut = highcut
        self.order = order
        self.fs = fs  
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def band_pass_filter(self):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        y = lfilter(b, a, self.data)
        return y
    
    def hanning_window(self):
        win_sizw = len(self.data)
        window = np.hanning(win_sizw)
        win_filter_data = window*self.data
        return win_filter_data 

    def fft(self):
        n = len(self.data)    
        k = np.arange(n)
        T = n/self.fs
        freq = k/T 
        freq = freq[range(int(n/2))]
        Y = np.fft.fft(self.data)/n 
        Y = Y[range(int(n/2))]
        return Y 
    
    def stft(self):
        y = librosa.stft(self.data, n_fft=128, hop_length=64, win_length=128)
        return y 
