import os, os.path

clip_duration = 10.0
sr = 32000
ref_db = -55
n_soundscapes = 2000
random_seed = 2023


sr = 32000
seg_sec = 10
n_window = 2048
hop_size = 255
n_mels = 128
mel_f_min = 0.
mel_f_max = 16000.
max_len_seconds = 10.

# pitch_shift = ("uniform", -3, 3)
syn_folder = os.path.join("/home/fumchin/data/bsed_20/dataset", "SYN")
bg_folder = os.path.join("/home/fumchin/data/bsed_20/dataset", "SYN", "background")
fg_folder = os.path.join("/home/fumchin/data/bsed_20/dataset", "SYN", "foreground")

syn_preprocess_folder = os.path.join(syn_folder, "preprocess")