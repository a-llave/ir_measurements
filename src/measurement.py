"""
Description: Record raw measurement

Author: Adrien Llave - CentraleSupelec
Date: 16/10/2018

Version: 1.0

Date    | Auth. | Vers.  |  Comments
18/10/16  ALl     1.0       Initialization

"""

import pyaudio
import time
import numpy as np
import src.pkg.RTAudioProc as rt
import wave
import matplotlib.pyplot as plt
import os


def record_measurement(metadata):

    # ----------------------------------------------------------------
    def callback(data_inp, frame_count, time_info, status):
        # ---- RECORD MEASUREMENT ------------------------------------------------------
        recording_x.append(data_inp)
        # ---- READ STIMULUS ------------------------------------------------------
        data_out = stimulus.readframes()
        data_out = rt.encode(data_out)
        return data_out, pyaudio.paContinue

    # ----------------------------------------------------------------
    # ---- PATH
    # STIMULUS
    path_stim = '../resources/stimulus'
    filename_stim = path_stim + '/' + metadata.stim_s

    # RAW MEASUREMENT
    path_raw_output = metadata.folder_path_s+'/raw_output'
    os.makedirs(path_raw_output, exist_ok=True)
    if metadata.iter_n == -1:
        filename_raw_output = path_raw_output + '/' + metadata.meas_name_s + '_raw.wav'
    else:
        filename_raw_output = path_raw_output + '/' + metadata.meas_name_s + '_' + str(metadata.iter_n) + '_raw.wav'

    # --- PARAMETERS
    nb_channels = 2

    #
    p = pyaudio.PyAudio()
    nb_buffsamp = 50
    resolution = pyaudio.paInt16

    # ---- LOAD STIMULUS ------------------------------------------------------
    print('----------------- LOAD STIMULUS -----------------')
    print('Stimulus path: %s' % filename_stim)
    wv = wave.open(filename_stim, 'rb')
    samp_freq = wv.getframerate()
    nb_frames = wv.getnframes()
    duration = nb_frames / samp_freq

    # ADAPT CHANNEL NUMBER
    data_bin = wv.readframes(nb_frames)
    stim = rt.decode(data_bin, wv.getnchannels())
    stim = np.repeat(stim[:, 0][:, np.newaxis], nb_channels, axis=1)
    stimulus = rt.AudioCallback(stim, nb_buffsamp)

    print('Sampling frequency: %.2f Hz' % samp_freq)
    print('Stimulus duration: %.2f s' % duration)
    print('Number of channels: %i' % nb_channels)
    print('Number of samples per buffer: %i' % nb_buffsamp)
    print('Frame duration: %.2f ms' % (nb_buffsamp/samp_freq*1000))

    # ---- STREAM ------------------------------------------------------
    stream = p.open(format=resolution,
                    channels=nb_channels,
                    rate=samp_freq,
                    output=True,
                    input=True,
                    input_device_index=metadata.device_inp_id,
                    output_device_index=metadata.device_out_id,
                    frames_per_buffer=nb_buffsamp,
                    stream_callback=callback)
    recording_x = []
    stream.start_stream()

    time.sleep(duration)

    stream.stop_stream()
    stream.close()
    p.terminate()
    wv.close()

    # ---- SAVE RAW MEASUREMENTS ------------------------------------------------------
    print('----------------- SAVE RAW MEASUREMENTS -----------------')
    print('PATH: '+filename_raw_output)
    waveFile = wave.open(filename_raw_output, 'wb')
    waveFile.setnchannels(nb_channels)
    waveFile.setsampwidth(p.get_sample_size(resolution))
    waveFile.setframerate(samp_freq)
    waveFile.writeframes(b''.join(recording_x))
    waveFile.close()

    # ---- LOAD RAW MEASUREMENT ------------------------------------------------------
    waveFile = wave.open(filename_raw_output, 'rb')
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

    # ---- PLOT ------------------------------------------------------
    plt.figure(1)
    plt.plot(stim[:, 0], 'b', label='Stimulus')
    plt.plot(measurement_raw[:, 0], 'r', label='Measurement')
    plt.legend()
    # plt.axis([unique_snr[0], unique_snr[nb_snr-1], 0, 100])
    plt.ylabel('Magnitude')
    plt.xlabel('Samples')
    plt.draw()

    return
