import numpy as np
import nlpaug.augmenter.audio as naa


def crop_audio(data, sr):
    aug = naa.CropAug(sampling_rate=sr)
    return aug.augment(data)


def add_mask(data, sr):
    aug = naa.MaskAug(sampling_rate=sr, mask_with_noise=False)
    return aug.augment(data)


def add_noise(data, sr):
    aug = naa.NoiseAug(noise_factor=0.03)
    return aug.augment(data)


def add_pitch(data, sr):
    aug = naa.PitchAug(sampling_rate=sr, pitch_range=(2, 3))
    return aug.augment(data)


def add_shift(data, sr):
    aug = naa.ShiftAug(sampling_rate=sr)
    return aug.augment(data)


def add_speed(data, sr):
    aug = naa.SpeedAug()
    return aug.augment(data)


def add_vltk(data, sr):
    aug = naa.VtlpAug(sampling_rate=sr)
    return aug.augment(data)


def freq_mask(data):
    v = data.shape[0]
    f = np.random.randint(80)
    f0 = np.random.randint(v - f)
    data[f0: f0 + f] = 0
    return data


def time_mask(data):
    t = np.random.randint(80)
    time_range = t + 20
    t0 = np.random.randint(time_range - t)
    data[:, t0:t0+t] = 0
    return data

