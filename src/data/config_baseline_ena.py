import logging
import math
import os, os.path
import torch
# path related
dataset_root = "/home/fumchin/data/bsed_20/dataset/ENA"
feature_dir = os.path.join(dataset_root, "preprocess_02_015")
annotation_dir = os.path.join(feature_dir, "annotation")

# train_feature_dir = os.path.join(dataset_root, "train_preprocess")
# train_annotation_dir = os.path.join(train_feature_dir, "annotation")
train_unlabeled_feature_dir = os.path.join(dataset_root, "train_unlabeled_preprocess_quarter_02_015")
train_unlabeled_annotation_dir = os.path.join(train_unlabeled_feature_dir, "annotation")

train_weak_feature_dir = os.path.join(dataset_root, "train_weak_preprocess_quarter_02_015")
train_weak_annotation_dir = os.path.join(train_weak_feature_dir, "annotation")

val_feature_dir = os.path.join(dataset_root, "val_preprocess_quarter_02_015")
val_annotation_dir = os.path.join(val_feature_dir, "annotation")

# dataset_root = "/home/fumchin/data/bsed_20/dataset/ENA"
# feature_dir = os.path.join(dataset_root, "preprocess_02")
# annotation_dir = os.path.join(feature_dir, "annotation")

# # train_feature_dir = os.path.join(dataset_root, "train_preprocess")
# # train_annotation_dir = os.path.join(train_feature_dir, "annotation")
# train_unlabeled_feature_dir = os.path.join(dataset_root, "train_unlabeled_preprocess_02")
# train_unlabeled_annotation_dir = os.path.join(train_unlabeled_feature_dir, "annotation")

# train_weak_feature_dir = os.path.join(dataset_root, "train_weak_preprocess_02")
# train_weak_annotation_dir = os.path.join(train_weak_feature_dir, "annotation")

# val_feature_dir = os.path.join(dataset_root, "val_preprocess_02")
# val_annotation_dir = os.path.join(val_feature_dir, "annotation")


syn_feature_dir = os.path.join("/home/fumchin/data/bsed_20/dataset/SYN", "preprocess")

synth_dataset_root = "/home/fumchin/data/bsed_20/dataset/SYN"
synth_feature_dir = os.path.join(synth_dataset_root, "preprocess")
synth_annotation_dir = os.path.join(synth_dataset_root, "annotation")
synth_audio_dir = os.path.join(synth_dataset_root, "NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV")

# audio
# mel dim (1255, 128)
# target dim (313, 30)
sr = 32000
seg_sec = 10
n_window = 2048
hop_size = 255
n_mels = 128
mel_f_min = 0.
mel_f_max = 16000.
max_len_seconds = 10.

max_frames = math.ceil(max_len_seconds * sr / hop_size)
pooling_time_ratio = 4

noise_snr = 30
median_window_s = 0.45
out_nb_frames_1s = sr / hop_size / pooling_time_ratio # 4 for pooling_time_ratio
median_window_s_classwise = [0.45, 0.45, 0.45, 0.45, 0.45, 2.7, 2.7, 2.7, 0.45, 2.7] # [0.3, 0.9, 0.9, 0.3, 0.3, 2.7, 2.7, 2.7, 0.9, 2.7] 
median_window = [max(int(item * out_nb_frames_1s), 1) for item in median_window_s_classwise]
# max_frames = math.ceil(max_len_seconds * sample_rate / hop_size)


in_memory = True
in_memory_unlab = False
num_workers = 12
batch_size = 12

# model_name = "CRNN_0323_fpn_ada_official_default_lr_advw_10"
# model_name = "only_fix_detach_ada_no_prediction_"
# model_name = "CRNN_fpn_scmt_test"

model_name = '0601_Quarter_3000_02_015_CRNN_fpn_all_ena'
# model_name = '0523_Quarter_3000_02_015_CRNN_fpn_mt_resPL_cdan_clip_stage2_seperate_SGD_lr_test'
# model_name = 'test'
# at_model_name = '0429_Quarter_3000_02_at_system'
at_model_name = 'resNet_at'
test_model_name = "CRNN_fpn_adlr"
only_syn = True
n_epoch = 300 #, variance after 100 may be too large
n_epoch_rampup = 50
n_epoch_rampdown = 80
dataset_random_seed = 1215


randon_layer_dim = 8192
Rf = torch.randn(80128, randon_layer_dim)
Rg = torch.randn(6260, randon_layer_dim)

checkpoint_epochs = 1
save_best = True
early_stopping = None # 20
es_init_wait = 50  # es for early stopping
adjust_lr = False
max_learning_rate = 0.001 #0.001  # Used if adjust_lr is True
default_learning_rate = 0.001#0.001  # Used if adjust_lr is False
max_consistency_cost = 1

# bird list
bird_list = \
[
    "EATO", "WOTH", "BCCH", "BTNW", "TUTI", 
    "NOCA", "REVI", "AMCR", "BLJA", "OVEN", 
    "COYE", "BGGN", "SCTA", "AMRE", "KEWA", 
    "BHCO", "BHVI", "HETH", "RBWO", "BAWW"
]
terminal_level = logging.INFO