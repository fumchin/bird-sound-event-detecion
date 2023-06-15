# -*- coding: utf-8 -*-
import argparse
import os.path as osp

import torch, torch.nn
# from psds_eval import PSDSEval
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# from data_utils.DataLoad import DataLoadDf
# from data_utils.Desed import DESED
# # from evaluation_measures import compute_sed_eval_metrics, psds_score, get_predictions
# from evaluation_measures import psds_score, get_predictions, \
#     compute_psds_from_operating_points, compute_metrics, get_f_measure_by_class
from utilities.utils import to_cuda_if_available, generate_tsv_wav_durations, meta_path_to_audio_dir
from evaluation_measures import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics, get_f_measure_by_class
from data.dataload import SYN_Dataset, ENA_Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
from utilities.ManyHotEncoder import ManyHotEncoder
from data.Transforms import get_transforms
from utilities.Logger import create_logger
from utilities.Scaler import Scaler, ScalerPerAudio
from models.CRNN import CRNN, Predictor, CRNN_fpn
import data.config as cfg
import pdb
import collections
import os
import os, os.path
import pdb
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.feature.inverse

from scipy.io.wavfile import write
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, silhouette_score

logger = create_logger(__name__)
torch.manual_seed(2023)


def _load_crnn(state, model_name="model", use_fpn=False):
    crnn_args = state[model_name]["args"]
    crnn_kwargs = state[model_name]["kwargs"]
    if use_fpn:
        crnn = CRNN_fpn(*crnn_args, **crnn_kwargs)
    else:
        crnn = CRNN(*crnn_args, **crnn_kwargs)

    # if "ADDA" in model_path:
    # if not use_fpn:
    # for key in list(expe_state["model"]["state_dict"].keys()):
    #     if 'cnn.' in key:
    #         expe_state["model"]["state_dict"][key.replace('cnn.', 'cnn.cnn.')] = expe_state["model"]["state_dict"][key]
    #         del expe_state["model"]["state_dict"][key]
    if not use_fpn:        
        for key in list(state["model"]["state_dict"].keys()):
            if 'cnn.' in key:
                state["model"]["state_dict"][key.replace('cnn.', 'cnn.cnn.')] = state["model"]["state_dict"][key]
                del state["model"]["state_dict"][key]

    crnn.load_state_dict(state[model_name]["state_dict"])
    crnn.eval()
    crnn = to_cuda_if_available(crnn)
    logger.info("Model loaded at epoch: {}".format(state["epoch"]))
    logger.info(crnn)
    return crnn


def _load_scaler(state):
    scaler_state = state["scaler"]
    type_sc = scaler_state["type"]
    if type_sc == "ScalerPerAudio":
        scaler = ScalerPerAudio(*scaler_state["args"])
    elif type_sc == "Scaler":
        scaler = Scaler()
    else:
        raise NotImplementedError("Not the right type of Scaler has been saved in state")
    scaler.load_state_dict(state["scaler"]["state_dict"])
    return scaler


def _load_state_vars(state, median_win=None, use_fpn=False, use_predictor=False):
    # pred_df = gtruth_df.copy()
    # Define dataloader
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    # scaler = _load_scaler(state)
    crnn = _load_crnn(state, use_fpn=use_fpn)

    pooling_time_ratio = state["pooling_time_ratio"]
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    if median_win is None:
        median_win = state["median_window"]
    if use_predictor == False:
        return {
            "model": crnn,
            # "strong_dataloader": strong_dataloader_ind,
            # "weak_dataloader": weak_dataloader_ind,
            "pooling_time_ratio": pooling_time_ratio,
            "many_hot_encoder": many_hot_encoder,
            "median_window": median_win,
            "predictor": None
        }
    else:
        predictor_args = state["model_p"]["args"]
        predictor_kwargs = state["model_p"]["kwargs"]
        predictor = Predictor(**predictor_kwargs)
        predictor.load_state_dict(expe_state["model_p"]["state_dict"])
        predictor.eval()

        predictor = to_cuda_if_available(predictor)
        return {
            "model": crnn,
            # "strong_dataloader": strong_dataloader_ind,
            # "weak_dataloader": weak_dataloader_ind,
            "pooling_time_ratio": pooling_time_ratio,
            "many_hot_encoder": many_hot_encoder,
            "median_window": median_win,
            "predictor": predictor
        }


def get_variables(args):
    model_pth = args.model_path
    gt_fname, ext = osp.splitext(args.groundtruth_tsv)
    median_win = args.median_window
    meta_gt = args.meta_gt
    gt_audio_pth = args.groundtruth_audio_dir
    use_fpn = args.use_fpn
    use_predictor = args.use_predictor

    if meta_gt is None:
        meta_gt = gt_fname + "_durations" + ext

    if gt_audio_pth is None:
        gt_audio_pth = meta_path_to_audio_dir(gt_fname)
        # Useful because of the data format
        if "validation" in gt_audio_pth:
            gt_audio_pth = osp.dirname(gt_audio_pth)
    print(os.getcwd())
    groundtruth = pd.read_csv(args.groundtruth_tsv, sep="\t")
    if osp.exists(meta_gt):
        meta_dur_df = pd.read_csv(meta_gt, sep='\t')
        if len(meta_dur_df) == 0:
            meta_dur_df = generate_tsv_wav_durations(gt_audio_pth, meta_gt)
    else:
        meta_dur_df = generate_tsv_wav_durations(gt_audio_pth, meta_gt)

    return model_pth, median_win, gt_audio_pth, groundtruth, meta_dur_df, use_fpn, use_predictor




def visualization(ADA_synth_feature, ADA_real_feature, ADA_path=None, No_ADA_path=None, sample_num=1000):
    print("start visualization")

    ADA_tsne_path = os.path.join(ADA_path, 'tsne(frame).npy')

    # if not os.path.exists(ADA_tsne_path):
    # synth_sample_index = np.random.choice(ADA_synth_feature.shape[0], sample_num)
    real_sample_index = np.random.choice(ADA_real_feature.shape[0], sample_num)

    # ADA_synth_feature_sample = ADA_synth_feature
    ADA_real_feature_sample = ADA_real_feature
   
    # ADA_feature = np.concatenate((ADA_synth_feature_sample, ADA_real_feature_sample), axis=0)
    ADA_feature = ADA_real_feature_sample
    ADA_feature = TSNE(n_components=3, perplexity=1).fit_transform(ADA_feature)
  
    np.save(ADA_tsne_path, ADA_feature)
    # print(ADA_recon.shape)
    ADA_feature = np.load(ADA_tsne_path)

    # ADA_synth_feature_sample = ADA_feature[:int(ADA_synth_feature_sample.shape[0])]
    # ADA_real_feature_sample = ADA_feature[int(ADA_synth_feature_sample.shape[0]):]

    time1_feature = ADA_feature[0]
    time2_feature = ADA_feature[1]
    time3_feature = ADA_feature[2]
    # ADA_synth_label = ['synth'] * int(ADA_synth_feature_sample.shape[0])
    # ADA_real_label = ['real'] * int(ADA_real_feature_sample.shape[0])
    time1_label = ['time1']
    time2_label = ['time2']
    time3_label = ['time3']
    ADA_label = np.array(time1_label + time2_label + time3_label)
    # s_score = silhouette_score(ADA_feature, ADA_label)
    # print('silhouette score: ', s_score)
    plt.figure()
    # plt.scatter(ADA_synth_feature_sample[:,0], ADA_synth_feature_sample[:,1], alpha=0.6, s=10, marker='x', label='synth')
    #plt.scatter(df['x'], df['y'], alpha=0.6, s=10, c=df['label'], label='synth')
    # sns.scatterplot(x="x", y="y", hue="label", palette="deep", linewidth=0, data=df_real, s=10)
    # plt.scatter(ADA_synth_feature_sample[:,0], ADA_synth_feature_sample[:,1], alpha=0.6, s=10, marker='x', label='synth')
    # plt.scatter(ADA_real_feature_sample[:,0], ADA_real_feature_sample[:,1], alpha=0.3, s=10, c='r', label='real')
    plt.scatter(time1_feature[:,0], time1_feature[:,1], alpha=0.6, s=10, marker='x', label='time1')
    plt.scatter(time2_feature[:,0], time2_feature[:,1], alpha=0.3, s=10, c='r', label='time2')
    plt.scatter(time3_feature[:,0], time3_feature[:,1], alpha=0.3, s=10, c='g', label='time3')
    # transfer back to audio
    plt.legend()
    # plt.title('syn vs real')
    # plt.xlim(-60, 80)
    # plt.ylim(-90, 90)
    plt.axis('off')
    plt.savefig(os.path.join(ADA_path, 'syn vs real(frame).png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", '--model_path', type=str, required=None,
                        help="Path of the model to be evaluated")
    parser.add_argument("-g", '--groundtruth_tsv', type=str, required=None,
                        help="Path of the groundtruth tsv file")

    # Not required after that, but recommended to defined
    parser.add_argument("-mw", "--median_window", type=int, default=None,
                        help="Nb of frames for the median window, "
                             "if None the one defined for testing after training is used")

    # Next groundtruth variable could be ommited if same organization than DESED dataset
    parser.add_argument('--meta_gt', type=str, default=None,
                        help="Path of the groundtruth description of feat_filenames and durations")
    parser.add_argument("-ga", '--groundtruth_audio_dir', type=str, default=None,
                        help="Path of the groundtruth filename, (see in config, at dataset folder)")
    parser.add_argument("-s", '--save_predictions_path', type=str, default=None,
                        help="Path for the predictions to be saved (if needed)")

    # Dev
    parser.add_argument("-n", '--nb_files', type=int, default=None,
                        help="Number of files to be used. Useful when testing on small number of files.")
    # Use fpn
    parser.add_argument("-fpn", '--use_fpn', action="store_true",
                    help="Whether to use CRNN_fpn architecture.")  
    # Use predictor
    parser.add_argument("-pd", '--use_predictor', default=True,
                    help="Whether to use label predictor.")
    # Dir to save embedding feature
    parser.add_argument("-sf", '--saved_feature_dir', action="store_true",
                        help="Path for the embedded features to be saved (if needed), Ex. embedded_feature/ADDA_with_synthetic_clipD_meanteacher_ISP, which kinds of data (weak/strong/synthetic) would be automatically detected")

    f_args = parser.parse_args()
    # Get variables from f_args
    # median_window, use_fpn, use_predictor = get_variables(f_args)
    test_model_name = "0505_Quarter_3000_02_015_CRNN_fpn_scmt"

    median_window = f_args.median_window
    use_fpn = f_args.use_fpn
    use_predictor = f_args.use_predictor
    model_path = os.path.join("/home/fumchin/data/bsed_20/src/stored_data", test_model_name, "model", "baseline_best")
    sf = f_args.saved_feature_dir
    if sf:
        saved_path = os.path.join("/home/fumchin/data/bsed_20/src/stored_data", test_model_name, "mix")
        
    

    # Model
    expe_state = torch.load(model_path)
    add_axis_conv = 0

    many_hot_encoder = ManyHotEncoder(cfg.bird_list, n_frames=cfg.max_frames // cfg.pooling_time_ratio)
    encod_func = many_hot_encoder.encode_strong_df
    # dataset = DESED(base_feature_dir=osp.join(cfg.workspace, "dataset", "features"), compute_log=False)
    # scaler = _load_scaler(state)
    # transforms = get_transforms(cfg.max_frames, None, add_axis_conv, noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    
    scaler = Scaler()
    # transforms_scaler = get_transforms(cfg.max_frames, add_axis=add_axis_conv)
    # train_scaler_dataset = ENA_Dataset(preprocess_dir=cfg.train_feature_dir, encod_func=encod_func, transform=transforms_scaler, compute_log=True)
    # syn_scaler_dataset = SYN_Dataset(preprocess_dir=cfg.synth_feature_dir, encod_func=encod_func, transform=transforms_scaler, compute_log=True)
    # scaler.calculate_scaler(ConcatDataset([train_scaler_dataset, syn_scaler_dataset])) 
    # transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv,
    #                                   noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    # real_dataset = ENA_Dataset(preprocess_dir=cfg.train_feature_dir, encod_func=encod_func, transform=transforms, compute_log=True)
    # syn_dataset = ENA_Dataset(preprocess_dir=cfg.synth_feature_dir, encod_func=encod_func, transform=transforms, compute_log=True)
    
    
    scaler_val = Scaler()
    # transforms_scaler = get_transforms(cfg.max_frames, add_axis=add_axis_conv)
    # val_scaler_dataset = ENA_Dataset(preprocess_dir=cfg.val_feature_dir, encod_func=encod_func, transform=transforms_scaler, compute_log=True)
    # scaler_val.calculate_scaler(val_scaler_dataset) 
    transforms_valid = get_transforms(cfg.max_frames, None, add_axis_conv,
                                      noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    val_test_dir = "/home/fumchin/data/bsed_20/dataset/SYN_test/preprocess_mix"
    val_dataset = ENA_Dataset(preprocess_dir=val_test_dir, encod_func=encod_func, transform=transforms_valid, compute_log=True)
    
    
    # transforms = get_transforms(cfg.max_frames, None, add_axis_conv,
    #                             noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    # val_dataset = ENA_Dataset(preprocess_dir=cfg.val_feature_dir, encod_func=encod_func, transform=transforms, compute_log=True)
    # train_data, val_data = train_test_split(dataset, random_state=cfg.dataset_random_seed, train_size=0.5)
    
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    params = _load_state_vars(expe_state, median_window, use_fpn, use_predictor)

    # Preds with only one value
    train_saved_feature_dir = os.path.join(saved_path, "train")
    # syn_saved_feature_dir = os.path.join(saved_path, "syn")
    # val_saved_feature_dir = os.path.join(saved_path, "val")

    if not os.path.exists(train_saved_feature_dir):
        os.makedirs(train_saved_feature_dir)

    # if not os.path.exists(syn_saved_feature_dir):
    #     os.makedirs(syn_saved_feature_dir)
    
    # if not os.path.exists(val_saved_feature_dir):
    #     os.makedirs(val_saved_feature_dir)

    if use_fpn:
        train_predictions, train_labels_df, train_durations_validation = get_predictions(params["model"], val_dataloader,
                                            params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                            median_window=params["median_window"],
                                            save_predictions=os.path.join(train_saved_feature_dir, 'pred.csv'),
                                            predictor=params["predictor"], fpn=True, saved_feature_dir=train_saved_feature_dir)
        