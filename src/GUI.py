"""
Description: Graphical User Interface

Author: Adrien Llave - CentraleSupelec
Date: 16/10/2018

Version: 1.0

Date    | Auth. | Vers.  |  Comments
18/10/16  ALl     1.0       Init

"""

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
from tkinter import filedialog
import os
import fnmatch
import pyaudio
from src.measurement import record_measurement
from src.raw2ir import raw2ir


# ============= CLASSES ==================
class MeasurementMetadata:
    def __init__(self, stim_s,
                 folder_path_s, meas_name_s, iter_n,
                 device_inp_id, device_out_id):

        self.stim_s = stim_s
        self.folder_path_s = folder_path_s
        self.meas_name_s = meas_name_s
        self.device_inp_id = device_inp_id
        self.device_out_id = device_out_id
        self.iter_n = iter_n


#  ============= FUNCTIONS ==================
def run_measurement():
    metadata = update_setup()
    print('========= RECORD MEASUREMENT ==================')
    record_measurement(metadata)
    if check_iter_tkv.get():
        spin_iter_tkv.set(str(metadata.iter_n+1))
    print('========= PROCESS RAW MEASUREMENT TO IMPULSE RESPONSE ==================')
    raw2ir(metadata)
    return


def test_setup():
    metadata = update_setup()
    print('--------------------------')
    print('Measurement metadata:')
    for key, value in metadata.__dict__.items():
        if isinstance(value, str):
            print('>> '+key + ': ' + value)
        else:
            print('>> '+key+': '+str(value))
    print('--------------------------')
    return


def update_setup():
    stim_s = stim_tkv.get()
    device_inp_s = device_inp_tkv.get()
    device_out_s = device_out_tkv.get()
    folder_path_s = folder_path_tkv.get()
    meas_name_s = meas_name_tkv.get()
    if check_iter_tkv.get:
        iter_n = int(spin_iter.get())

    device_inp_id = 1000
    device_out_id = 1000
    for api_id in range(p.get_host_api_count()):
        for device_id in range(p.get_device_count()):
            tmp_var = p.get_device_info_by_host_api_device_index(api_id, device_id)
            if tmp_var['name'] == device_inp_s and tmp_var['maxInputChannels'] != 0:
                device_inp_id = p.get_device_info_by_host_api_device_index(api_id, device_id)['index']
            if tmp_var['name'] == device_out_s and tmp_var['maxOutputChannels'] != 0:
                device_out_id = p.get_device_info_by_host_api_device_index(api_id, device_id)['index']

    metadata = MeasurementMetadata(stim_s,
                                   folder_path_s, meas_name_s, iter_n,
                                   device_inp_id, device_out_id)

    return metadata


def list_stim():
    """
    Return listing of wav file in stimulus folder
    :return:
    """
    list_of_files = os.listdir('../resources/stimulus')
    pattern = "*.wav"
    stim_name_list = [0] * len(list_of_files)
    for ii, entry in enumerate(list_of_files):
        if fnmatch.fnmatch(entry, pattern):
            stim_name_list[ii] = entry
    return stim_name_list


def browse():
    global folder_path_tkv
    tmp_path = filedialog.askdirectory()
    folder_path_tkv.set(tmp_path)
    return


def list_host_api():
    nb_host_api = p.get_host_api_count()
    host_api_name_list = [0] * nb_host_api
    for api_id in range(nb_host_api):
        # print('======================')
        # print('HOST API: ' + str(api_id))
        # print(p.get_host_api_info_by_index(api_id))
        host_api_name_list[api_id] = p.get_host_api_info_by_index(api_id)['name']
    return host_api_name_list


def list_device_inp():
    api_id = 0
    nb_device = p.get_device_count()
    device_name_list = []
    for device_id in range(nb_device):
        # print('-------------------')
        # print('Device :'+str(device_id))
        # print(p.get_device_info_by_index(device_id))
        if p.get_device_info_by_host_api_device_index(api_id, device_id)['maxInputChannels'] != 0:
            device_name_list.append(p.get_device_info_by_host_api_device_index(api_id, device_id)['name'])

    return device_name_list


def list_device_out():
    api_id = 0
    nb_device = p.get_device_count()
    device_name_list = []
    for device_id in range(nb_device):
        # print('-------------------')
        # print('Device :'+str(device_id))
        # print(p.get_device_info_by_index(device_id))
        if p.get_device_info_by_host_api_device_index(api_id, device_id)['maxOutputChannels'] != 0:
            device_name_list.append(p.get_device_info_by_host_api_device_index(api_id, device_id)['name'])
    return device_name_list


def on_combo_configure(event):
    font = tkfont.nametofont(str(event.widget.cget('font')))
    width = font.measure(device_out_name_list[0] + "0") - event.width
    style = ttk.Style()
    style.configure('TCombobox', postoffset=(0, 0, width, 0))
    return


def iterate_measurement():
    return


# ============= DEFINE GUI ==================
root = tk.Tk()

p = pyaudio.PyAudio()

# ============= DEFINITION VAR ==================
iter_n = -1

# ================= FRAME 1 =================
Frame1 = tk.LabelFrame(root, text="Measurement")
Frame1.grid(row=0, column=0)

#  ---- ROW 0
stim_label = tk.Label(Frame1, text='Stimulus:')
stim_label.grid(row=0, column=0)

stim_name_list = list_stim()
stim_tkv = tk.StringVar()
combo_stim = ttk.Combobox(Frame1, textvariable=stim_tkv, values=stim_name_list)
combo_stim.set(stim_name_list[0])
combo_stim.grid(row=0, column=1)

#  ---- ROW 1
folder_label = tk.Label(Frame1, text='folder path:')
folder_label.grid(row=1, column=0)

folder_path_tkv = tk.StringVar()
champ_label = tk.Label(Frame1, textvariable=folder_path_tkv)
champ_label.grid(row=1, column=1)

button_browse = tk.Button(Frame1, text="Browse", command=browse)
button_browse.grid(row=1, column=2)

#  ---- ROW 2
meas_name_label = tk.Label(Frame1, text='Measurement name:')
meas_name_label.grid(row=2, column=0)

meas_name_tkv = tk.StringVar()
meas_name_entry = tk.Entry(Frame1, textvariable=meas_name_tkv)
meas_name_entry.grid(row=2, column=1)

#  ---- ROW 3
iter_label = tk.Label(Frame1, text='Iterate measurement:')
iter_label.grid(row=3, column=0)

check_iter_tkv = tk.IntVar()
check_iter = tk.Checkbutton(Frame1, variable=check_iter_tkv)
check_iter.grid(row=3, column=1)

spin_iter_tkv = tk.StringVar()
spin_iter = tk.Spinbox(Frame1, from_=0, to=100, width=5, textvariable=spin_iter_tkv)
spin_iter.grid(row=3, column=2)

#  ---- ROW 4
button_run = tk.Button(Frame1, text="Run", command=run_measurement)
button_run.grid(row=4, column=1)

button_test = tk.Button(Frame1, text="Test", command=test_setup)
button_test.grid(row=4, column=2)

# ================= FRAME 2 =================
Frame2 = tk.LabelFrame(root, text="Device")
Frame2.grid(row=0, column=1)

#  ---- ROW 0
host_api_label = tk.Label(Frame2, text='Host API:')
host_api_label.grid(row=0, column=0)

host_api_name_list = list_host_api()
host_api_tkv = tk.StringVar()
combo_host_api = ttk.Combobox(Frame2, textvariable=host_api_tkv, values=host_api_name_list)
combo_host_api.set(host_api_name_list[0])
combo_host_api.grid(row=0, column=1)

#  ---- ROW 1
device_api_label = tk.Label(Frame2, text='Input device:')
device_api_label.grid(row=1, column=0)
device_inp_name_list = list_device_inp()
device_inp_tkv = tk.StringVar()
combo_device_inp = ttk.Combobox(Frame2, textvariable=device_inp_tkv, values=device_inp_name_list)
combo_device_inp.set(device_inp_name_list[0])
combo_device_inp.grid(row=1, column=1)

#  ---- ROW 1
device_api_label = tk.Label(Frame2, text='Output device:')
device_api_label.grid(row=2, column=0)

device_out_name_list = list_device_out()
device_out_tkv = tk.StringVar()
combo_device_out = ttk.Combobox(Frame2, textvariable=device_out_tkv, values=device_out_name_list)
combo_device_out.bind('<Configure>', on_combo_configure)
combo_device_out.set(device_out_name_list[0])
combo_device_out.grid(row=2, column=1)

# ================= ROOT =================
button_quit = tk.Button(root, text="Quit", command=root.quit)
button_quit.grid(row=1, column=0)

root.mainloop()
