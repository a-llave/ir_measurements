"""
Description: Convert raw measurement to impulse responses

Author: Adrien Llave - CentraleSupelec
Date: 16/10/2018

Version: 1.0

Date    | Auth. | Vers.  |  Comments
18/10/16  ALl     1.0       Initialization

"""


import numpy as np
import src.pkg.RTAudioProc as rt
import wave
import matplotlib.pyplot as plt
import os


def raw2ir(metadata):
    # ----------------------------------------------------------------
    # ---- PATH
    # STIMULUS
    path_stim = '../resources/stimulus'
    filename_stim = path_stim + '/' + 'one_sweep_48k.wav'
    print('Stimulus path: %s' % filename_stim)

    # RAW MEASUREMENT
    path_raw_output = metadata.folder_path_s+'/raw_output'
    if metadata.iter_n == -1:
        filename_raw_output = path_raw_output + '/' + metadata.meas_name_s + '_raw.wav'
    else:
        filename_raw_output = path_raw_output + '/' + metadata.meas_name_s + '_' + str(metadata.iter_n) + '_raw.wav'

    # IR OUTPUT
    path_ir_output = metadata.folder_path_s+'/ir_output'
    os.makedirs(path_ir_output, exist_ok=True)
    if metadata.iter_n == -1:
        filename_ir_output = path_ir_output + '/' + metadata.meas_name_s + '_ir.wav'
    else:
        filename_ir_output = path_ir_output + '/' + metadata.meas_name_s + '_' + str(metadata.iter_n) + '_ir.wav'

    # --- PARAMETERS
    nb_samp_ir_targ = 256
    nb_channels = 2

    # ---- LOAD RAW MEASUREMENT ------------------------------------------------------
    print('----------------- LOAD RAW MEASUREMENTS -----------------')
    print('PATH: ' + filename_raw_output)
    waveFile = wave.open(filename_raw_output, 'rb')
    samp_freq = waveFile.getframerate()
    nb_frames = waveFile.getnframes()
    measurement_raw = rt.decode(waveFile.readframes(nb_frames), nb_channels)
    waveFile.close()
    # FADE I/O
    meas_fadei_duration = 0.1
    meas_fadeo_duration = 0.1
    nb_samp_meas_fadei = int(meas_fadei_duration * samp_freq)
    nb_samp_meas_fadeo = int(meas_fadeo_duration * samp_freq)
    samples_meas_fadei_v = np.linspace(0, np.pi, nb_samp_meas_fadei)
    samples_meas_fadeo_v = np.linspace(0, np.pi, nb_samp_meas_fadeo)
    fadei_v = (1 - np.cos(samples_meas_fadei_v))/2
    fadeo_v = (1 + np.cos(samples_meas_fadeo_v))/2
    fadei_m = np.repeat(fadei_v[:, np.newaxis], nb_channels, axis=1)
    fadeo_m = np.repeat(fadeo_v[:, np.newaxis], nb_channels, axis=1)
    nb_samp_meas = measurement_raw.shape[0]
    measurement_raw[0: nb_samp_meas_fadei, :] = np.multiply(fadei_m, measurement_raw[0: nb_samp_meas_fadei, :])
    measurement_raw[nb_samp_meas-nb_samp_meas_fadeo: nb_samp_meas, :] = np.multiply(fadeo_m, measurement_raw[nb_samp_meas-nb_samp_meas_fadeo: nb_samp_meas, :])

    # ---- DECONVOLUTION ------------------------------------------------------
    # ---- LOAD STIMULUS
    wv = wave.open(filename_stim, 'rb')
    stimulus_m = rt.decode(wv.readframes(nb_frames), nb_channels)
    stimulus_inv_m = np.flipud(stimulus_m)
    # ---- DECONV
    ir_raw_m = np.zeros((nb_samp_meas+stimulus_inv_m.shape[0]-1, nb_channels))
    for ch in range(0, nb_channels):
        ir_raw_m[:, ch] = np.convolve(measurement_raw[:, ch], stimulus_inv_m[:, ch])

    # ---- CLEAN ------------------------------------------------------
    # REMOVE PRE-DELAY
    idx_max_v = np.argmax(np.abs(ir_raw_m), axis=0)  # FIND MAX PER CHANNELS
    idx_min = int(np.min(idx_max_v) - 0.02 * nb_samp_ir_targ)
    ir_cut_m = ir_raw_m[idx_min:ir_raw_m.shape[0], :]
    # FADE I/O
    idx_max_v = np.argmax(np.abs(ir_cut_m), axis=0)  # FIND MAX PER CHANNELS
    idx_min = np.min(idx_max_v)
    nb_samp_ir = ir_cut_m.shape[0]
    nb_samp_ir_fadei = int(0.95 * idx_min)
    nb_samp_ir_fadeo = int(0.3 * nb_samp_ir_targ)
    samples_ir_fadei_v = np.linspace(0, np.pi/2, nb_samp_ir_fadei)
    samples_ir_fadeo_v = np.linspace(0, np.pi/2, nb_samp_ir_fadeo)
    fadei_v = (1 - np.cos(samples_ir_fadei_v))**4
    fadeo_v = (1 + np.cos(samples_ir_fadeo_v))/2
    fadei_m = np.repeat(fadei_v[:, np.newaxis], nb_channels, axis=1)
    fadeo_m = np.repeat(fadeo_v[:, np.newaxis], nb_channels, axis=1)
    ir_cut_old_m = np.copy(ir_cut_m)
    ir_cut_m[0: nb_samp_ir_fadei, :] = np.multiply(fadei_m, ir_cut_m[0: nb_samp_ir_fadei, :])
    ir_cuto_m = ir_cut_m[0:nb_samp_ir_targ, :]
    ir_cuto_m[nb_samp_ir_targ-nb_samp_ir_fadeo: nb_samp_ir_targ, :] = np.multiply(fadeo_m, ir_cuto_m[nb_samp_ir_targ-nb_samp_ir_fadeo: nb_samp_ir_targ, :])

    tf_cuto = np.zeros(ir_cuto_m.shape)
    for ch in range(0, nb_channels):
        tf_cuto[:, ch] = 20 * np.log10(np.abs(np.fft.fft(ir_cuto_m[:, ch])))

    # ---- PLOT ------------------------------------------------------
    color_v = ['b', 'r', 'g', 'y']
    # for ch in range(0, nb_channels):
    #     plt.plot(ir_cut_m[:, ch], color_v[ch])
    plt.figure(2)
    plt.plot(ir_cut_old_m[:, 0], 'b')
    plt.plot(ir_cut_m[:, 0], 'r')
    plt.plot(ir_cuto_m[:, 0], 'g')
    plt.title('IR')
    # plt.axis([unique_snr[0], unique_snr[nb_snr-1], 0, 100])
    plt.ylabel('Magnitude')
    plt.xlabel('Samples')
    plt.show()

    freq_v = np.linspace(0, samp_freq, nb_samp_ir_targ)
    for ch in range(0, nb_channels):
        plt.semilogx(freq_v, tf_cuto[:, ch], 'g')
    plt.title('TF')
    plt.grid()
    plt.axis([100, 20000, np.min(tf_cuto), np.max(tf_cuto)])
    plt.ylabel('Magnitude (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.draw()

    # ---- SAVE IR ------------------------------------------------------
    print('----------------- SAVE CLEAN IR -----------------')
    print('PATH: ' + filename_ir_output)
    nframes = ir_cuto_m.shape[0]
    nchannels = ir_cuto_m.shape[1]
    sampwidth = 3
    waveFile = wave.open(filename_ir_output, 'wb')
    waveFile.setnchannels(nchannels)
    waveFile.setsampwidth(sampwidth)
    waveFile.setframerate(samp_freq)
    waveFile.setnframes(nframes)
    output = rt.encode(ir_cuto_m)
    waveFile.writeframes(output)
    waveFile.close()
