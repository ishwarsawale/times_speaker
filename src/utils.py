# Third Party
import librosa
import data_aug
import  os
import numpy as np

# ===============================================
#       code from Arsha for loading data.
# ===============================================


def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    print(path)
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

def graph_spectrogram(data, rate, name):
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=384, NFFT=512)
    ax.axis('off')
    fig.savefig(f'{name}.png', dpi=300, frameon='false')


def create_train_data(wav,win_length=400, sr=16000,
                      hop_length=160, n_fft=512, spec_len=250, mode='train'):
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    train_data = (spec_mag - mu) / (std + 1e-5)
    return train_data



def load_data_aug(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    all_data = []
    wav = load_wav(path, sr=sr, mode=mode)
    spectro_grams = []

    wav_crop = data_aug.crop_audio(wav, sr)
    wav_mask = data_aug.add_mask(wav, sr)
    wav_noise = data_aug.add_noise(wav, sr)
    wav_pitch = data_aug.add_pitch(wav, sr)
    wav_shift = data_aug.add_shift(wav, sr)
    wav_speed = data_aug.add_speed(wav, sr)
    wav_vltk = data_aug.add_vltk(wav, sr)
    all_data.append(wav)
    all_data.append(wav_crop)
    all_data.append(wav_mask)
    all_data.append(wav_noise)
    all_data.append(wav_pitch)
    all_data.append(wav_shift)
    all_data.append(wav_speed)
    all_data.append(wav_vltk)

    for wav in all_data:
        train_data = create_train_data(wav)
        spectro_grams.append(train_data)

    file_name = os.path.basename(path)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    linear_spect = data_aug.freq_mask(linear_spect)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    train_data = (spec_mag - mu) / (std + 1e-5)
    spectro_grams.append(train_data)

    # graph_spectrogram(train_data, sr, f'{file_name}_val_freq')

    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    linear_spect = data_aug.time_mask(linear_spect)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    train_data = (spec_mag - mu) / (std + 1e-5)
    spectro_grams.append(train_data)

    # graph_spectrogram(train_data, sr, f'{file_name}_val_time')

    return spectro_grams
