"""
Description:

Author: Adrien Llave - CentraleSupelec
Date: 04/04/2018

Version: 1.0

Date    | Auth. | Vers.  |  Comments
18/04/04  ALl     1.0       Initialization

"""

import pyaudio
import time
from time import sleep
import numpy as np
import src.pkg.RTAudioProc as rt
import wave
import matplotlib.pyplot as plt


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
filename_stim = path_stim + '/' + 'one_sweep_48k.wav'
print('Stimulus path: %s' % filename_stim)

# RAW MEASUREMENT
path_raw_output = '../resources/raw_output'
filename_raw_output = path_raw_output + '/' + 'raw_measurements.wav'

# IR OUTPUT
path_ir_output = '../resources/ir_output'
filename_ir_output = path_ir_output + '/' + 'ir_measurements.wav'

# --- PARAMETERS
nb_samp_ir_targ = 256
nb_channels = 2

#
p = pyaudio.PyAudio()
nb_buffsamp = 50
resolution = pyaudio.paInt16

# ---- LOAD STIMULUS ------------------------------------------------------
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

# for ii in range(0, 10):
#     print(ii)
#     print(p.get_host_api_info_by_index(ii))
#     for jj in range(0, p.get_device_count()):
#         print(p.get_device_info_by_host_api_device_index(ii, jj))

stream = p.open(format=resolution,
                channels=nb_channels,
                rate=samp_freq,
                output=True,
                input=True,
                # input_device_index=16,
                # output_device_index=16,
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

# ---- DECONVOLUTION ------------------------------------------------------
wv = wave.open(filename_stim, 'rb')
stimulus_m = rt.decode(wv.readframes(nb_frames), nb_channels)
stimulus_inv_m = np.flipud(stimulus_m)
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
# plt.plot(stimulus_m[:, 0], 'b', label='Stimulus')
# plt.plot(measurement_raw[:, 0], 'r', label='Measurement')
# plt.legend()
# # plt.axis([unique_snr[0], unique_snr[nb_snr-1], 0, 100])
# plt.ylabel('Magnitude')
# plt.xlabel('Samples')
# plt.show()

color_v = ['b', 'r', 'g', 'y']
# for ch in range(0, nb_channels):
#     plt.plot(ir_cut_m[:, ch], color_v[ch])
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
plt.show()

# ---- SAVE IR ------------------------------------------------------
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
