"""
Description: class and functions needed for the real time audio process

Author: Adrien Llave - CentraleSupelec
Date: 01/04/2018

Version: 3.0

Date    | Auth. | Vers.  |  Comments
18/03/28  ALl     1.0       Initialization
18/03/30  ALl     2.0       Bug fix in Butterworth class, remove initial click due to bad conditioning
18/04/01  ALl     3.0       Minor bug fix:  - DMA compensation gain
                                            - DMA hybrid Lowpass filter cutoff frequency high to optimize WNG

"""

import numpy as np
import scipy.signal as spsig


# ======================================================================================================
# =================== FUNCTIONS ========================================================================
# ======================================================================================================


def decode(in_data, channels):
    """
    Convert a byte stream into a 2D numpy array with
    shape (chunk_size, channels)

    Samples are interleaved, so for a stereo stream with left channel
    of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...], the output
    is ordered as [L0, R0, L1, R1, ...]
    """
    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    result = np.fromstring(in_data, dtype=np.int16)
    chunk_length = int(len(result) / channels)
    assert chunk_length == int(chunk_length)
    result = np.reshape(result, (chunk_length, channels))
    result = result.astype(np.float32)
    result = result / np.power(2.0, 16)
    return result


def encode(signal):
    """
    Convert a 2D numpy array into a byte stream for PyAudio

    Signal should be a numpy array with shape (chunk_size, channels)
    """
    signal = signal * np.power(2.0, 16)
    signal = signal.astype(np.int16)
    interleaved = signal.flatten()

    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    out_data = interleaved.astype(np.int16).tostring()
    return out_data


def voice_activity_detector(signal, threshold=0.0):
    nb_xzero = 0
    nb_sample = len(signal)
    vad_b = False
    for nn in range(1, nb_sample):
        if not np.sign(signal[nn]) == np.sign(signal[nn-1]):
            nb_xzero += 1
    xzero_rate = nb_xzero / nb_sample
    if xzero_rate < threshold:
        vad_b = True
    return vad_b

# ======================================================================================================
# =================== CLASSES ==========================================================================
# ======================================================================================================


class Codec:
    def __init__(self, type=np.int16):
        self.type = type
        if self.type is np.int16:
            self.res_n = 16
        elif self.type is np.int32:
            self.res_n = 32
        elif self.type is np.int32:
            self.res_n = 0
        else:
            print('ERROR')

    def encode(self, signal):
        """
            Convert a 2D numpy array into a byte stream for PyAudio

            Signal should be a numpy array with shape (chunk_size, channels)
            """
        signal = signal * np.power(2.0, self.res_n)
        signal = signal.astype(self.type)
        interleaved = signal.flatten()
        out_data = interleaved.astype(self.type).tostring()
        return out_data

    def decode(self, in_data, channels):
        """
        Convert a byte stream into a 2D numpy array with
        shape (chunk_size, channels)

        Samples are interleaved, so for a stereo stream with left channel
        of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...], the output
        is ordered as [L0, R0, L1, R1, ...]
        """
        result = np.fromstring(in_data, dtype=self.type)
        chunk_length = int(len(result) / channels)
        assert chunk_length == int(chunk_length)
        result = np.reshape(result, (chunk_length, channels))
        result = result.astype(np.float32)
        result = result / np.power(2.0, self.res_n)
        return result


class AudioCallback:
    def __init__(self, sig_m, nb_buffsamp, backward_b=False):
        self.sig_m = sig_m
        self.nb_samples = self.sig_m.shape[0]
        self.nb_channels = self.sig_m.shape[1]
        self.nb_buffsamp = nb_buffsamp
        self.backward_b = backward_b
        if not self.backward_b:
            self.marker_in = 0
        else:
            self.marker_in = self.nb_samples

        return

    def readframes(self):
        id_inp = self.marker_in - int(self.backward_b)*self.nb_buffsamp
        id_out = self.marker_in+self.nb_buffsamp - int(self.backward_b)*self.nb_buffsamp
        buffer_m = self.sig_m[id_inp:id_out, :][::(-1)**int(self.backward_b), :]
        self.marker_in = self.marker_in + (-1)**int(self.backward_b) * self.nb_buffsamp
        return buffer_m


class Compressor:
    """
    ================================
    Real time dynamic range compressor
    ================================

    """
    def __init__(self, nb_bufsamp,
                 nb_channels,
                 thrsh=0,
                 ratio=1,
                 time_attack=0.005,
                 time_release=0.05,
                 knee_width=1,
                 fs_f=48000,
                 bypass=False):
        """
        Constructor
        """

        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = nb_channels
        self.gain_db = np.zeros((self.nb_bufsamp, self.nb_channels))
        self.env_prev = np.zeros((self.nb_bufsamp, self.nb_channels))
        self.thrsh = thrsh
        self.ratio = ratio
        self.coeff_attack = np.exp(-1/(time_attack*fs_f))
        self.coeff_release = np.exp(-1/(time_release*fs_f))
        self.kneeWidth = knee_width
        self.bypass = bypass

    def process(self, npdata_in):
        """
        Process dynamic range compression
        """
        if not self.bypass:
            # GAIN COMPUTER
            inp_gr = 20 * np.log10(np.absolute(npdata_in) + np.finfo(float).eps)
            out_gr = np.zeros(npdata_in.shape)
            out_env = np.zeros(npdata_in.shape)
            out_gr[2 * (inp_gr - self.thrsh) < - self.kneeWidth] = inp_gr[2 * (inp_gr - self.thrsh) < - self.kneeWidth]
            out_gr[2 * np.absolute(inp_gr - self.thrsh) <= self.kneeWidth] = inp_gr[2 * np.absolute(inp_gr - self.thrsh) <= self.kneeWidth] \
                                                                                    + (1 / self.ratio - 1) \
                                                                                    * (inp_gr[2 * np.absolute(inp_gr - self.thrsh) <= self.kneeWidth]
                                                                                    - self.thrsh + self.kneeWidth / 2) ** 2 \
                                                                                    / (2 * self.kneeWidth)
            out_gr[2 * (inp_gr - self.thrsh) > self.kneeWidth] = self.thrsh + (inp_gr[2 * (inp_gr - self.thrsh) > self.kneeWidth] - self.thrsh) / self.ratio
            # PEAK DETECTOR
            for cc in range(0, self.nb_channels):
                inp_env = inp_gr - out_gr
                if inp_env[0, cc] > self.env_prev[self.nb_bufsamp-1, cc]:
                    out_env[0, cc] = self.coeff_attack * self.env_prev[self.nb_bufsamp-1, cc] + (1 - self.coeff_attack) * inp_env[0, cc]
                else:
                    out_env[0, cc] = self.coeff_release * self.env_prev[self.nb_bufsamp-1, cc]

                for nn in range(1, self.nb_bufsamp):
                    if inp_env[nn, cc] > out_env[nn - 1, cc]:
                        out_env[nn, cc] = self.coeff_attack * out_env[nn - 1, cc] + (1 - self.coeff_attack) * inp_env[nn, cc]
                    else:
                        out_env[nn, cc] = self.coeff_release * out_env[nn - 1, cc]

            # BACK UP
            self.env_prev = out_env
            # GAIN REDUCTION
            self.gain_db = -out_env
            npdata_out = npdata_in * np.power(10, self.gain_db / 20)

        else:
            npdata_out = npdata_in

        return npdata_out


class Convolution:
    def __init__(self, nb_bufsamp, nb_channels, bypass=False):
        """
        Constructor
        """
        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = nb_channels
        self.data_overlap = np.zeros((self.nb_bufsamp-1, self.nb_channels), dtype=np.float32)
        self.bypass = bypass

    def process(self, sig1, sig2):
        """
        Process the convolution between sig 1 and sig2
        """
        if not self.bypass:
            # SHAPE
            sig1_shape = sig1.shape
            sig2_shape = sig2.shape
            # ASSERTION NCHANNELS
            nsample_ft = sig1_shape[0] + sig2_shape[0] - 1
            # FFT PARTY
            sig1_ft = np.fft.fft(sig1, nsample_ft, 0)
            sig2_ft = np.fft.fft(sig2, nsample_ft, 0)
            sig_out_ft = np.multiply(sig1_ft, sig2_ft)
            # COME BACK TO TIME DOMAIN
            sig_out = np.real(np.fft.ifft(sig_out_ft, nsample_ft, 0))
            # SPLIT OUTPUT AND OVERLAP TMP
            output = sig_out[0:sig1_shape[0], :]
            overlap_tmp = sig_out[sig1_shape[0]:sig_out.shape[0], :]
            # OVERLAP ADD
            if sig1_shape[0] > self.data_overlap.shape[0]:  # N > L-1
                data_overlap_2actualframe = np.concatenate((self.data_overlap, np.zeros((sig1_shape[0] - self.data_overlap.shape[0], self.data_overlap.shape[1]))), axis=0)
                overlap_prev = np.zeros(self.data_overlap.shape)
            else:  # L-1 >= N
                data_overlap_2actualframe = self.data_overlap[0:sig1_shape[0]]
                overlap_prev = self.data_overlap[sig1_shape[0]:self.data_overlap.shape[0]]
                overlap_prev = np.concatenate((overlap_prev, np.zeros((sig1_shape[0], self.data_overlap.shape[1]))), axis=0)

            output = output + data_overlap_2actualframe
            # OVERLAP UPDATE
            self.data_overlap = overlap_tmp + overlap_prev

        else:
            output = sig1

        return output


class ConvolutionIR:
    def __init__(self, nb_bufsamp, ir_m=np.concatenate((np.array([1.0])[:, np.newaxis], np.zeros((127, 1))),
                                                       axis=0), bypass=False):
        """
        Constructor
        """
        self.nb_bufsamp = nb_bufsamp
        self.nb_channels = ir_m.shape[1]
        self.IR_m = ir_m
        self.nsample_ft = self.nb_bufsamp + self.IR_m.shape[0] - 1
        self.TF_m = np.fft.fft(ir_m, self.nsample_ft, 0)
        self.data_overlap = np.zeros((self.IR_m.shape[0]-1, self.nb_channels), dtype=np.float32)
        self.bypass = bypass

    def process(self, sig1):
        """
        Process the convolution between sig 1 and IR
        """
        if not self.bypass:
            # CHECK
            if len(sig1.shape) == 1:
                sig1 = sig1[:, np.newaxis]

            assert sig1.shape[1] == self.nb_channels, 'signal input and the IR must have the same number of ' \
                                                      'columns i.e. the same number of channels'
            # CONVOLUTION
            sig_out = np.zeros((self.nsample_ft, self.nb_channels))
            for ch in range(0, self.nb_channels):
                sig_out[:, ch] = np.convolve(sig1[:, ch], self.IR_m[:, ch])
            # SPLIT OUTPUT AND OVERLAP TMP
            output = sig_out[0:self.nb_bufsamp, :]
            overlap_tmp = sig_out[self.nb_bufsamp:self.nsample_ft, :]
            # OVERLAP ADD
            if self.nb_bufsamp > self.data_overlap.shape[0]:  # N > L-1
                data_overlap_2actualframe = np.concatenate((self.data_overlap, np.zeros((self.nb_bufsamp - self.data_overlap.shape[0], self.nb_channels))), axis=0)
                overlap_prev = np.zeros(self.data_overlap.shape)
            else:  # L-1 >= N
                data_overlap_2actualframe = self.data_overlap[0:self.nb_bufsamp]
                overlap_prev = self.data_overlap[self.nb_bufsamp:self.data_overlap.shape[0]]
                overlap_prev = np.concatenate((overlap_prev, np.zeros((self.nb_bufsamp, self.data_overlap.shape[1]))), axis=0)

            output = output + data_overlap_2actualframe
            # OVERLAP UPDATE FOR NEXT FRAME
            self.data_overlap = overlap_tmp + overlap_prev

        else:
            output = sig1

        return output

    def update_ir(self, ir_m):
        """
        Update impulse response
        """
        # TODO: remove discontinuity due to data_overlap size actualization when M < N or P < L
        old_size_ir_n = self.IR_m.shape[0]
        old_nb_ch_n = self.IR_m.shape[1]
        self.IR_m = ir_m
        self.nb_channels = self.IR_m.shape[1]
        self.nsample_ft = self.nb_bufsamp + self.IR_m.shape[0] - 1
        self.TF_m = np.fft.fft(ir_m, self.nsample_ft, 0)

        # OVERLAP VECTOR SIZE ACTUALIZATION
        # RESIZE NB CHANNELS
        if self.IR_m.shape[1] == old_nb_ch_n:  # M = N
            # print('M = N')
            pass
        elif self.IR_m.shape[1] > old_nb_ch_n:  # M > N
            self.data_overlap = np.concatenate((self.data_overlap, np.zeros((old_size_ir_n-1, self.nb_channels-old_nb_ch_n))), axis=1)
        elif self.IR_m.shape[1] < old_size_ir_n:  # M < N
            self.data_overlap = self.data_overlap[:, 0:self.nb_channels]
        else:
            print('PROBLEM IN UPDATE IR')

        # RESIZE IR SIZE
        if self.IR_m.shape[0] == old_size_ir_n:  # P = L
            # print('P = L')
            pass
        elif self.IR_m.shape[0] > old_size_ir_n:  # P > L
            self.data_overlap = np.concatenate((self.data_overlap, np.zeros((self.IR_m.shape[0]-old_size_ir_n, self.nb_channels))), axis=0)
        elif self.IR_m.shape[0] < old_size_ir_n:  # P < L
            self.data_overlap = self.data_overlap[0:self.IR_m.shape[0] - 1, :]
        else:
            print('PROBLEM IN UPDATE IR')

        return


class Butterworth:
    def __init__(self, nb_buffsamp, samp_freq=48000, nb_channels=1, type_s='low', cut_freq=100., order_n=1, bypass=False):
        """
        Constructor
        """
        # GENERAL
        self.bypass = bypass
        self.nb_buffsamp = nb_buffsamp
        self.samp_freq = samp_freq
        self.nb_channels = nb_channels
        self.xfade_b = False
        # FILTER
        self.type_s = type_s
        self.cut_freq = cut_freq
        self.order_n = order_n
        wn = 2 * self.cut_freq / self.samp_freq
        self.filt_b, self.filt_a = spsig.butter(self.order_n, wn, self.type_s, analog=False)
        self.filt_b_old = self.filt_b
        self.filt_a_old = self.filt_a
        self.previous_info = spsig.lfiltic(self.filt_b, self.filt_a, np.zeros(len(self.filt_b)-1,))
        self.previous_info = np.repeat(self.previous_info[:, np.newaxis], self.nb_channels, axis=1)
        self.previous_info_old = self.previous_info
        # FADE I/O
        self.fadeinp_v = np.repeat(np.sin(np.linspace(0, np.pi/2, self.nb_buffsamp))[:, np.newaxis]**2,
                                   self.nb_channels, axis=1)
        self.fadeout_v = np.repeat(np.cos(np.linspace(0, np.pi/2, self.nb_buffsamp))[:, np.newaxis]**2,
                                   self.nb_channels, axis=1)

    def process(self, sig1):
        """
        Process the filtering on sig1
        :param sig1: matrix [nb_samples X nb_channels]
        :return output: matrix [nb_samples X nb_channels]
        """
        if not self.bypass:
            # CHECK
            if len(sig1.shape) == 1:
                sig1 = sig1[:, np.newaxis]

            assert sig1.shape[1] == self.nb_channels, 'signal input and the IR must have the same number of ' \
                                                      'columns i.e. the same number of channels'

            output, self.previous_info = spsig.lfilter(self.filt_b,
                                                       self.filt_a,
                                                       sig1,
                                                       axis=0,
                                                       zi=self.previous_info)
            # XFADE WHEN CHANGE FILTER PARAM
            if self.xfade_b:
                xoutput, _ = spsig.lfilter(self.filt_b_old,
                                           self.filt_a_old,
                                           sig1,
                                           axis=0,
                                           zi=self.previous_info_old)

                output = self.fadeout_v * xoutput + self.fadeinp_v * output
                self.xfade_b = False
        else:
            output = sig1

        return output

    def update_filter(self, type_s='low', cut_freq=100, order_n=1):
        """
        Update filter
        :param type_s: 'low', 'high', 'band'. Default: 'low'
        :param cut_freq: cut-off frequency. Default: 100 Hz
        :param order_n: filter order. Default: 1
        :return:
        """

        self.type_s = type_s
        if self.type_s is 'band':
            assert len(cut_freq) == 2, 'You must specify low and high cut-off frequency for a band-pass'
        self.cut_freq = cut_freq
        self.order_n = order_n
        wn = 2 * self.cut_freq / self.samp_freq
        old_len_prev = self.previous_info.shape[0]
        self.filt_a_old = self.filt_a
        self.filt_b_old = self.filt_b
        self.previous_info_old = self.previous_info
        self.filt_b, self.filt_a = spsig.butter(self.order_n, wn, self.type_s, analog=False)
        new_len_prev = max(len(self.filt_a), len(self.filt_b)) - 1
        if old_len_prev > new_len_prev:
            self.previous_info = self.previous_info[0:new_len_prev, :]
            self.xfade_b = True
        elif old_len_prev == new_len_prev:
            pass
        elif old_len_prev < new_len_prev:
            self.previous_info = np.concatenate((self.previous_info,
                                                 np.zeros((new_len_prev - old_len_prev,
                                                          self.nb_channels))),
                                                axis=0)
        else:
            print('UNEXPECTED')

        return


class FilterBank:
    def __init__(self, nb_buffsamp, samp_freq, nb_channels, frq_band_v, bypass=False):
        # GENERAL
        self.bypass = bypass
        self.nb_buffsamp = nb_buffsamp
        self.samp_freq = samp_freq
        self.nb_channels = nb_channels
        # FILTER BANK
        self.frq_band_v = frq_band_v
        self.nb_band = self.frq_band_v.shape[0]
        self.frq_cut_v = np.zeros((self.nb_band-1,))
        for id_band in range(0, self.nb_band-1):
            self.frq_cut_v[id_band] = np.sqrt(self.frq_band_v[id_band] * self.frq_band_v[id_band+1])
        self.Filters = []
        # FIRST LOW PASS
        self.Filters.append(Butterworth(nb_buffsamp=self.nb_buffsamp,
                                        samp_freq=samp_freq,
                                        nb_channels=nb_channels,
                                        type_s='low',
                                        cut_freq=self.frq_cut_v[0],
                                        order_n=1)
                            )
        # BAND PASS
        for id_band in range(1, self.nb_band-1):
            self.Filters.append(Butterworth(nb_buffsamp=self.nb_buffsamp,
                                            samp_freq=self.samp_freq,
                                            nb_channels=self.nb_channels,
                                            type_s='band',
                                            cut_freq=np.array([self.frq_cut_v[id_band-1], self.frq_cut_v[id_band]]),
                                            order_n=1)
                                )
        # LAST HIGH PASS
        self.Filters.append(Butterworth(nb_buffsamp=self.nb_buffsamp,
                                        samp_freq=self.samp_freq,
                                        nb_channels=self.nb_channels,
                                        type_s='high',
                                        cut_freq=self.frq_cut_v[self.nb_band-2],
                                        order_n=1)
                            )

    def process(self, sig1):
        """
        Process the filtering on sig1
        :param sig1: matrix [nb_samples X nb_channels]
        :return output: matrix [nb_samples X nb_channels x nb_band]
        """
        if not self.bypass:
            sig_out = np.zeros((self.nb_buffsamp, self.nb_channels, self.nb_band))
            for id_band in range(0, self.nb_band):
                sig_out[:, :, id_band] = self.Filters[id_band].process(sig1)
        else:
            sig_out = sig1
        return sig_out


class BeamformerDAS:
    def __init__(self, nb_buffsamp,
                 ir_m=np.concatenate((np.array([1.0])[np.newaxis, :].T, np.zeros((1, 127))), axis=1)[np.newaxis, :].T,
                 adaptive_b=True,
                 bypass=False):
        """
        Constructor
        """
        # GENERAL
        self.bypass = bypass
        # PROCESS
        self.nb_buffsamp = nb_buffsamp
        self.nb_channels = ir_m.shape[1]
        self.IR_m = ir_m
        self.nb_ftsamp = int(np.power(2, np.ceil(np.log2(self.nb_buffsamp + self.IR_m.shape[0] - 1))))
        self.TF_m = np.fft.fft(ir_m, self.nb_ftsamp, 0)
        self.data_overlap = np.zeros((self.nb_ftsamp - self.nb_buffsamp, 1), dtype=np.float32)
        self.Wf = np.zeros((self.nb_channels, self.nb_ftsamp), dtype=complex)
        # FOR BEAMPATTERN PLOT
        self.nb_dir = 1
        self.coords_m = np.array([0, 0])
        self.pattern = np.zeros((self.nb_ftsamp, self.nb_dir))
        self.TF_by_dir = np.zeros((self.nb_ftsamp, self.nb_dir))
        # ADAPTIVE BEAMFORMER VAR
        self.adaptive_b = adaptive_b
        self.refresh = 30
        self.frame_count = 0
        self.cov_tot_m = np.repeat(np.eye(self.nb_channels)[:, :, np.newaxis], int(self.nb_ftsamp/2)+1, axis=2)
        self.inv_cov_tot_m = np.repeat(np.eye(self.nb_channels)[:, :, np.newaxis], int(self.nb_ftsamp/2)+1, axis=2)
        # FOR EXTERNAL PLOT
        self.buff_1 = np.zeros((self.nb_buffsamp,))
        self.buff_2 = np.zeros((self.nb_buffsamp,))
        self.overlap_1 = np.zeros((self.nb_buffsamp,))
        self.overlap = np.zeros((self.nb_buffsamp,))
        self.sigout_1 = np.zeros((self.nb_ftsamp,))
        self.sigout = np.zeros((self.nb_ftsamp,))

    def process(self, sig1):
        """
        Process the beamforming between sig 1 and IR
        """

        if not self.bypass:
            # CHECK
            assert sig1.shape[1] == self.nb_channels, 'signal input and the IR must have the same number of ' \
                                                        'columns i.e. the same number of channels'
            # FFT PARTY
            sig1_ft = np.fft.fft(sig1, self.nb_ftsamp, 0)

            # UPDATE FILTER IF NEEDED
            if self.adaptive_b and self.frame_count == self.refresh:
                self.frame_count = 0
                self.compute_inv_cov_mat(sig1_ft)
                self.update_inv_filter()
            self.frame_count += 1
            # APPLY BEAMFORMING FILTER
            sig_out_ft = np.zeros((sig1_ft.shape[0], 1), dtype=complex)
            for freq in range(0, self.nb_ftsamp):
                xf = sig1_ft[freq, :]
                xf = xf.T
                yf = self.Wf[:, freq].dot(xf)
                sig_out_ft[freq] = yf.T
            # COME BACK TO TIME DOMAIN
            sig_out = np.real(np.fft.ifft(sig_out_ft, self.nb_ftsamp, 0))
            # SPLIT OUTPUT AND OVERLAP TMP
            output = sig_out[0:self.nb_buffsamp, :]
            overlap_tmp = sig_out[self.nb_buffsamp:self.nb_ftsamp]
            # OVERLAP ADD
            if self.nb_buffsamp > self.data_overlap.shape[0]:  # N > L-1
                data_overlap_2actualframe = np.concatenate((self.data_overlap,
                                                            np.zeros((self.nb_buffsamp - self.data_overlap.shape[0], 1))),
                                                           axis=0)
                overlap_prev = np.zeros(self.data_overlap.shape)
            else:  # L-1 >= N
                data_overlap_2actualframe = self.data_overlap[0:self.nb_buffsamp]
                overlap_prev = self.data_overlap[self.nb_buffsamp:self.data_overlap.shape[0]]
                overlap_prev = np.concatenate((overlap_prev,
                                               np.zeros((self.nb_buffsamp, self.data_overlap.shape[1]))),
                                              axis=0)
            output = output + data_overlap_2actualframe
            # OVERLAP UPDATE FOR NEXT FRAME
            self.data_overlap = overlap_tmp + overlap_prev

            # self.overlap_1 = self.overlap
            # self.overlap = data_overlap_2actualframe[:, 0]
            # self.sigout_1 = self.sigout
            # self.sigout = sig_out[:, 0]
        else:
            output = sig1

        # self.buff_2 = self.buff_1
        # self.buff_1 = output[:, 0]

        return output

    def update_ir(self, ir_m, normalize=True, mic_id=4):
        """
        Update impulse response
        """
        # TODO: remove discontinuity due to data_overlap size actualization when P < L
        old_size_ir_n = self.IR_m.shape[0]
        old_size_overlap_n = self.data_overlap.shape[0]
        old_nb_ch_n = self.IR_m.shape[1]
        self.IR_m = ir_m
        self.nb_channels = self.IR_m.shape[1]
        self.nb_ftsamp = int(np.power(2, np.ceil(np.log2(self.nb_buffsamp + self.IR_m.shape[0] - 1))))
        self.cov_tot_m = np.repeat(np.eye(self.nb_channels)[:, :, np.newaxis], self.nb_ftsamp, axis=2)
        self.inv_cov_tot_m = np.repeat(np.eye(self.nb_channels)[:, :, np.newaxis], self.nb_ftsamp, axis=2)
        self.TF_m = np.fft.fft(ir_m, self.nb_ftsamp, 0)
        self.Wf = np.zeros((self.nb_channels, self.nb_ftsamp), dtype=complex)
        # FOR EXTERNAL PLOT
        self.buff_1 = np.zeros((self.nb_buffsamp,))
        self.buff_2 = np.zeros((self.nb_buffsamp,))
        self.overlap_1 = np.zeros((self.nb_buffsamp,))
        self.overlap = np.zeros((self.nb_buffsamp,))
        self.sigout_1 = np.zeros((self.nb_ftsamp,))
        self.sigout = np.zeros((self.nb_ftsamp,))

        # NORMALIZE
        if normalize:
            self.TF_m = self.TF_m / np.repeat(self.TF_m[:, mic_id][:, np.newaxis], self.nb_channels, axis=1)

        self.update_inv_filter()

        # RESIZE IR SIZE
        if self.IR_m.shape[0] == old_size_ir_n:  # P = L
            pass
        elif self.IR_m.shape[0] > old_size_ir_n:  # P > L
            self.data_overlap = np.concatenate((self.data_overlap,
                                                np.zeros((self.nb_ftsamp-self.nb_buffsamp-self.data_overlap.shape[0], 1))),
                                               axis=0)
        elif self.IR_m.shape[0] < old_size_ir_n:  # P < L
            # self.data_overlap = self.data_overlap[0:self.IR_m.shape[0] - 1, :]
            self.data_overlap = self.data_overlap[0:self.nb_ftsamp-self.nb_buffsamp]
        else:
            print('PROBLEM IN UPDATE IR')

        return

    def update_inv_filter(self):
        """
        Update inverse filter after IR updating or new estimation
        :return:
        """
        # COMPUTE INVERSE FILTER
        for freq in range(0, int(self.nb_ftsamp/2)+1):
            steer_v = self.TF_m[freq, :][np.newaxis, :].T
            inv_cov_m = np.squeeze(self.inv_cov_tot_m[:, :, freq])
            denom = np.matmul(np.conjugate(steer_v.T), np.matmul(inv_cov_m, steer_v))
            self.Wf[:, freq] = np.squeeze(np.matmul(inv_cov_m, steer_v) / denom) \
                               * np.exp(1j * 2 * np.pi * freq / (self.nb_ftsamp - 1) * 0.1 * self.nb_ftsamp)

        self.Wf = np.conjugate(self.Wf)
        self.Wf = np.concatenate(
            (self.Wf[:, 0:int(self.nb_ftsamp/2)+1], np.conjugate(np.fliplr(self.Wf[:, 1:int(self.nb_ftsamp/2)]))),
            axis=1)

        # FADE I/O
        ir_tmp = np.real(np.fft.ifft(self.Wf))
        fadein_percent = 0.05
        fadeout_percent = 0.6
        ir_tmp[:, 0:int(fadein_percent * ir_tmp.shape[1])] = ir_tmp[:,
                                                             0:int(fadein_percent * ir_tmp.shape[1])] * np.repeat(
            np.linspace(0., 1., int(fadein_percent * ir_tmp.shape[1]))[np.newaxis, :], self.nb_channels, axis=0)
        ir_tmp[:, int(fadeout_percent * ir_tmp.shape[1]):ir_tmp.shape[1]] = ir_tmp[:,
                                                                            int(fadeout_percent * ir_tmp.shape[1]):
                                                                            ir_tmp.shape[1]] * np.repeat(
            np.linspace(1., 0., ir_tmp.shape[1] - int(fadeout_percent * ir_tmp.shape[1]))[np.newaxis, :],
            self.nb_channels, axis=0)

        self.Wf = np.fft.fft(ir_tmp, axis=1)
        return

    def compute_inv_cov_mat(self, sig1, alpha_f=0.9, eps_f=1e-8):
        """

        :param sig1: vector for 1 frequency [nb_channel x 1]
        :param alpha_f: update
        :param eps_f: loading factor for matrix inversion
        :return:
        """
        for freq in range(0, int(self.nb_ftsamp/2)+1):
            cov_m = self.cov_tot_m[:, :, freq]
            sig_tmp_v = np.abs(sig1[freq, :][np.newaxis, :].T)
            cov_m = alpha_f * np.squeeze(cov_m) + (1-alpha_f) * np.matmul(sig_tmp_v, sig_tmp_v.T)
            self.cov_tot_m[:, :, freq] = cov_m
            inv_cov_m = np.linalg.inv(cov_m + eps_f*np.eye(cov_m.shape[0]))
            self.inv_cov_tot_m[:, :, freq] = inv_cov_m

        return

    def load_data_for_beampattern(self, ir_by_dir, coords_m):
        self.nb_dir = ir_by_dir.shape[2]
        self.coords_m = coords_m

        assert self.nb_dir == self.coords_m.shape[0], \
            'Third dimension of ir_by_dir and first dimension of coords_m must agree'
        self.pattern = np.zeros((int(self.nb_ftsamp/2)+1, self.nb_dir))
        self.TF_by_dir = np.zeros((self.nb_ftsamp, self.nb_channels, self.nb_dir), dtype=complex)
        for dir_idx in range(0, self.nb_dir):
            self.TF_by_dir[:, :, dir_idx] = np.fft.fft(np.squeeze(ir_by_dir[:, :, dir_idx]), self.nb_ftsamp, axis=0)

        return

    def compute_beampattern(self):
        for dir_idx in range(0, self.nb_dir):
            for freq in range(0, int(self.nb_ftsamp/2)+1):
                self.pattern[freq, dir_idx] = np.abs(self.Wf[:, freq].dot(np.squeeze(self.TF_by_dir[freq, :, dir_idx])))

        return


class BeamformerDMA:
    """
    Order 1 DMA
    """
    def __init__(self, samp_freq, nb_buffsamp, nullangle_v=180., mic_dist=0.013,
                 freq_cutlp=100., freq_cuthp=80., bypass=False):
        """

        :param samp_freq:
        :param nb_buffsamp:
        :param nullangle_v:
        :param mic_dist:
        :param bypass:
        """
        # GENERAL
        self.samp_freq = samp_freq
        self.bypass = bypass
        self.nb_buffsamp = nb_buffsamp
        # ALGO
        self.nullangle_v = nullangle_v
        self.coeff_v = np.cos(np.deg2rad(self.nullangle_v)) / (np.cos(np.deg2rad(self.nullangle_v)) - 1)
        # FILTER
        self.freq_cutlp = freq_cutlp
        self.freq_cuthp = freq_cuthp
        # LOWPASS
        self.LPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=1,
                                    type_s='low',
                                    cut_freq=self.freq_cutlp,
                                    order_n=1
                                    )
        # HIGHPASS
        self.HPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=1,
                                    type_s='high',
                                    cut_freq=self.freq_cuthp,
                                    order_n=3
                                    )
        # COMPENSATION GAIN
        self.velocity = 343.
        self.mic_dist = mic_dist
        self.gain_dipole = np.sqrt(1 + (self.velocity / (self.mic_dist * 6 * self.freq_cutlp)) ** 2)

    def process(self, sig_inp):
        """

        :param sig_inp: input signal
        :return:
        """
        if not self.bypass:
            sig_dif = sig_inp[:, 0] - sig_inp[:, 1]  # DIFF
            sig_out = self.coeff_v * sig_inp[:, 0] \
                      + (1 - self.coeff_v) * self.LPFilter.process(sig_dif)[:, 0] * self.gain_dipole
            sig_out = self.HPFilter.process(sig_out)  # HIGHPASS FILTER
        else:
            sig_out = sig_inp[:, 0]

        return sig_out

    def define_nullangle(self, nullangle_v=180.):
        """

        :param nullangle_v: angle of the destructive constraint
        :return:
        """
        self.nullangle_v = nullangle_v
        self.coeff_v = np.cos(np.deg2rad(self.nullangle_v)) / (np.cos(np.deg2rad(self.nullangle_v)) - 1)
        return


class BeamformerDMA2:
    def __init__(self, samp_freq, nb_buffsamp, nullangle_v=np.array([180., 90.]),
                 mic_dist=0.013, freq_cutlp=100., freq_cuthp=80., bypass=False):
        """
        Constructor
        :param samp_freq: sampling frequency
        :param nb_buffsamp: buffer number of sample
        :param nullangle_v: angles of the destructive constrains
        (cardio: (180,90) ; hypercardio: (144,72) ; supercardio: (153,106) ; quadrupole: (135,45)
        :param mic_dist: distance between microphones
        :param bypass: Bypass
        """
        # GENERAL
        self.bypass = bypass
        self.samp_freq = samp_freq
        self.nb_buffsamp = nb_buffsamp
        # ALGO
        self.nullangle_v = nullangle_v
        self.mic_dist = mic_dist
        # CALL SUB DMA
        self.freq_cutlp = freq_cutlp
        self.freq_cuthp = freq_cuthp
        self.sub_dma_1 = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                       nullangle_v=self.nullangle_v[0], mic_dist=self.mic_dist,
                                       freq_cutlp=self.freq_cutlp)
        self.sub_dma_1.HPFilter.bypass = True
        self.sub_dma_2 = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                       nullangle_v=self.nullangle_v[0], mic_dist=self.mic_dist,
                                       freq_cutlp=self.freq_cutlp)
        self.sub_dma_2.HPFilter.bypass = True
        self.sub_dma_3 = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                       nullangle_v=self.nullangle_v[1], mic_dist=self.mic_dist,
                                       freq_cutlp=self.freq_cutlp, freq_cuthp=self.freq_cuthp)

    def process(self, sig_inp):
        """

        :param sig_inp: input signal
        :return:
        """
        if not self.bypass:
            sig_subout_1 = self.sub_dma_1.process(sig_inp[:, 0:2])
            sig_subout_2 = self.sub_dma_2.process(sig_inp[:, 1:3])
            sig_tmp = np.concatenate((sig_subout_1[:, np.newaxis], sig_subout_2[:, np.newaxis]), axis=1)
            sig_out = self.sub_dma_3.process(sig_tmp)
        else:
            sig_out = sig_inp[:, 0][:, np.newaxis]

        return sig_out

    def define_nullangle(self, nullangle_v):
        """

        :param nullangle_v: angle of the destructive constraint
        :return:
        """
        self.nullangle_v = nullangle_v
        self.sub_dma_1.define_nullangle(nullangle_v=nullangle_v[0])
        self.sub_dma_2.define_nullangle(nullangle_v=nullangle_v[0])
        self.sub_dma_3.define_nullangle(nullangle_v=nullangle_v[1])
        return


class BeamformerDMA15:
    def __init__(self, samp_freq, nb_buffsamp, nullangle_v=np.array([180., 180., 90.]),
                 mic_dist=0.013, freq_cut=800., freq_cutlp_dma2=300., bypass=False):
        # GENERAL
        self.bypass = bypass
        self.samp_freq = samp_freq
        self.nb_buffsamp = nb_buffsamp
        # ALGO
        self.nullangle_v = nullangle_v
        self.mic_dist = mic_dist
        # CALL SUB DMA
        self.sub_dma_1 = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                       nullangle_v=self.nullangle_v[0], mic_dist=2*self.mic_dist)
        self.freq_cutlp_dma2 = freq_cutlp_dma2
        self.sub_dma_2 = BeamformerDMA2(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                        nullangle_v=self.nullangle_v[1:3], mic_dist=self.mic_dist, freq_cutlp=self.freq_cutlp_dma2)
        # CROSSOVER FILTER
        self.velocity = 343.
        self.freq_cut = freq_cut
        # LOWPASS
        self.LPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=1,
                                    type_s='low',
                                    cut_freq=self.freq_cut,
                                    order_n=2
                                    )
        # HIGHPASS
        self.HPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=1,
                                    type_s='high',
                                    cut_freq=self.freq_cut,
                                    order_n=2
                                    )

    def process(self, sig_inp):
        """

        :param sig_inp: input signal
        :return:
        """
        if not self.bypass:
            sig_inp_1 = np.concatenate((sig_inp[:, 0][:, np.newaxis], sig_inp[:, 2][:, np.newaxis]), axis=1)
            sig_subout_1 = self.sub_dma_1.process(sig_inp_1)
            sig_subout_1_lp = self.LPFilter.process(sig_subout_1)
            sig_subout_2 = self.sub_dma_2.process(sig_inp)
            sig_subout_2_hp = self.HPFilter.process(sig_subout_2)
            sig_out = sig_subout_1_lp + sig_subout_2_hp

        else:
            sig_out = sig_inp[:, 0][:, np.newaxis]

        return sig_out


class DmaInteraural:
    def __init__(self, samp_freq, nb_buffsamp, nullangle_v=180., mic_dist=0.14, freq_cut=800., bypass=False):
        # GENERAL
        self.bypass = bypass
        self.samp_freq = samp_freq
        self.nb_buffsamp = nb_buffsamp
        # ALGO
        self.nullangle_v = nullangle_v
        self.mic_dist = mic_dist
        # CALL SUB DMA
        self.dma_l = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                   nullangle_v=self.nullangle_v, mic_dist=self.mic_dist)
        self.dma_r = BeamformerDMA(samp_freq=self.samp_freq, nb_buffsamp=self.nb_buffsamp,
                                   nullangle_v=self.nullangle_v, mic_dist=self.mic_dist)
        # CROSSOVER FILTER
        self.freq_cut = freq_cut
        # LOWPASS
        self.LPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=2,
                                    type_s='low',
                                    cut_freq=self.freq_cut,
                                    order_n=2
                                    )
        # HIGHPASS
        self.HPFilter = Butterworth(nb_buffsamp=self.nb_buffsamp,
                                    samp_freq=self.samp_freq,
                                    nb_channels=2,
                                    type_s='high',
                                    cut_freq=self.freq_cut,
                                    order_n=2
                                    )

    def process(self, sig_inp):
        """

        :param sig_inp: input signal
        :return:
        """
        if not self.bypass:

            sig_inp_l = sig_inp
            sig_inp_r = np.fliplr(sig_inp)

            sig_l = self.dma_l.process(sig_inp_l)
            sig_r = self.dma_r.process(sig_inp_r)
            sig_dma = np.concatenate((sig_l, sig_r), axis=1)

            sig_lp = self.LPFilter.process(sig_dma)
            sig_hp = self.HPFilter.process(sig_inp)

            sig_out = sig_lp + sig_hp

        else:
            sig_out = sig_inp

        return sig_out
