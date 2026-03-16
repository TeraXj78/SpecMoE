import numpy as np
#import random

import matplotlib.pyplot as plt
from scipy.signal import stft, istft



def stft_masking(
    x,
    sr=256,
    n_fft=512,
    noverlap=256,
    mask_ratio=0.5,
    mask_level=1.0,
    mask_type_ratio=[0.6, 0.3, 0.1],  # [freq, time, time-freq]
    global_seed=42,
    batch_idx=0,
    deterministic_per_sample=True,
    shared_mask_across_channels=True,
    visualize=False,
    device="cpu",
    band_bias=0.5,  # fraction of frequency-related masks constrained to EEG bands
    band_list=[
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 12.0),
        ("beta", 12.0, 30.0)
    ],
    band_probs=[0.3, 0.4, 0.2, 0.1],
):


    C, T = x.shape
    x_masked = np.zeros_like(x)
    mask_maps = []

    #b = 0

    # Deterministic RNG
    rng = np.random.RandomState(global_seed + batch_idx * 100 + b) if deterministic_per_sample else np.random

    # STFT per channel
    f, t_frames, Zxx_list = [], [], []
    for c in range(C):
        f_c, t_c, Zxx_c = stft(x[c, :], fs=sr, nperseg=n_fft, noverlap=noverlap)  
        f, t_frames = f_c, t_c
        Zxx_list.append(Zxx_c)
    Freqs, Times = Zxx_list[0].shape
    total_area = Freqs * Times

    # band bin indices
    band_bins = []
    for _, f_low, f_high in band_list:
        idx = np.where((f >= f_low) & (f <= f_high))[0]
        band_bins.append(idx)

    mask_total = np.ones((Freqs, Times))
    target_area = mask_ratio * total_area
    current_area = 0.0

    
    while current_area < target_area:
        mask_choice = rng.choice(["freq", "time", "tf"], p=mask_type_ratio)

        # Random centers by default
        f0 = rng.randint(0, Freqs)
        t0 = rng.randint(0, Times)

        # Possibly constrain frequency mask to EEG band
        if mask_choice in ["freq", "tf"] and rng.rand() < band_bias:
            if len(band_list) > 0:
                band_idx = rng.choice(len(band_list), p=np.array(band_probs) / np.sum(band_probs))
                bins = band_bins[band_idx]
                if bins.size > 0:
                    f0 = int(rng.choice(bins))

        # Generate Gaussian
        if mask_choice == "freq":
            sf = Freqs * 0.05
            G_freq = np.exp(-0.5 * ((np.arange(Freqs)[:, None] - f0) / sf) ** 2)
            G_time = np.ones((1, Times))
            G = G_freq * G_time

        elif mask_choice == "time":
            st = Times * 0.05
            G_freq = np.ones((Freqs, 1))
            G_time = np.exp(-0.5 * ((np.arange(Times)[None, :] - t0) / st) ** 2)
            G = G_freq * G_time

        else:  # "tf"
            sf = Freqs * 0.05
            st = Times * 0.05
            G = np.exp(-0.5 * ((np.arange(Freqs)[:, None] - f0) / sf) ** 2) * \
                np.exp(-0.5 * ((np.arange(Times)[None, :] - t0) / st) ** 2)

        mask = 1.0 - mask_level * G
        mask_total *= mask
        current_area = (1.0 - mask_total.mean()) * total_area

    mask_maps.append(mask_total)

    # Mask applied to each channel
    for c in range(C):
        masked_stft = Zxx_list[c] * mask_total if shared_mask_across_channels else Zxx_list[c] * mask_total.copy()
        _, x_rec = istft(masked_stft, fs=sr, nperseg=n_fft, noverlap=noverlap)
        x_masked[c, :len(x_rec)] = x_rec[:T]   # x_masked[b, c, :len(x_rec)] = x_rec[:T]

    # === Visualization ===
    if visualize == "True":
        b, c = 0, 0
        t_axis = np.arange(T) / sr

        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.title("Raw EEG signal")
        plt.plot(t_axis, x[c], color='steelblue', linewidth=0.8)       #  x_masked[b,c]
        plt.subplot(2, 1, 2)
        plt.title("Masked EEG signal")
        plt.plot(t_axis, x_masked[c], color='darkred', linewidth=0.8)  #  x_masked[b,c]
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

        orig_mag = np.log(np.abs(Zxx_list[c]) + 1e-6)
        masked_mag = np.log(np.abs(Zxx_list[c] * mask_maps[0]) + 1e-6)
        vmin, vmax = min(orig_mag.min(), masked_mag.min()), max(orig_mag.max(), masked_mag.max())

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(orig_mag, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title("Original STFT")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(masked_mag, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Masked STFT")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    # === Convert back to torch ===
    info = {
        "mask_ratio": mask_ratio,
        "mask_level": mask_level,
        "mask_type_ratio": mask_type_ratio,
        "global_seed": global_seed,
        "batch_idx": batch_idx,
        "band_bias": band_bias,
        "band_list": band_list,
        "band_probs": band_probs,
        "mask_maps": mask_maps,
        "stft_shape": (Freqs, Times),
    }

    return x_masked, info   #  [x_masked[:,i,:] for i in range(C)]



