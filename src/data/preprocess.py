import os, os.path
import glob
from glob import glob
import threading

import librosa
import pandas as pd
import numpy as np
import random
import shutil
from scipy.sparse.csgraph import connected_components


import config as cfg
import warnings
# from pandas.core.common import SettingWithCopyWarning
# warnings.filterwarnings('ignore', category=SettingWithCopyWarning)
def preprocess(audio, compute_log=False):
    ham_win = np.hamming(cfg.n_window)
    mel_min_max_freq = (cfg.mel_f_min, cfg.mel_f_max)
    spec = librosa.stft(
        audio,
        n_fft=cfg.n_window,
        hop_length=cfg.hop_size,
        window=ham_win,
        center=True,
        pad_mode='reflect'
    )
    
    mel_spec = librosa.feature.melspectrogram(
        S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
        sr=cfg.sr,
        n_mels=cfg.n_mels,
        fmin=mel_min_max_freq[0], 
        fmax=mel_min_max_freq[1],
        htk=False, 
        norm=None
    )
    
    if compute_log:
        mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
    mel_spec = mel_spec.T
    mel_spec = mel_spec.astype(np.float32)
    
    return mel_spec

def overlap(df, time):
    if(len(df.index) == 0):
        return df
    
    df_overlap = df.loc[(df["onset"] < time) & (df["offset"] > time)]
    df_non_overlap = df.loc[~((df["onset"] < time) & (df["offset"] > time))]
    
    df_overlap = pd.concat([df_overlap]*2, ignore_index=True)
    df_overlap = df_overlap.sort_values(by=['event_label', 'onset', 'offset'])
    
    for i in range(len(df_overlap.index)):
        if (i % 2) == 0:
            df_overlap.loc[i, "offset"] = time - 10**(-6)
        else:
            df_overlap.loc[i , "onset"] = time
    
    df_result = pd.concat([df_non_overlap, df_overlap], axis=0, ignore_index=True)
    
    return df_result

def same_event_label_overlap(df):
    if(len(df.index) == 0):
        return df
    # df = df.sort_values(by=['event_label', 'onset'])
    # result_df = None
    # event_label_list = df["event_label"].unique()
    # for count, current_event_label in enumerate(event_label_list):
    #     current_df = df.loc[df['event_label'] == current_event_label]
    #     current_df["group"]=(current_df["onset"]>current_df["offset"].shift().cummax()).cumsum()
    #     current_result=current_df.groupby("group").agg({"onset":"min", "offset": "max"}).reset_index()
    #     current_result["event_label"] = current_event_label
    #     current_result = current_result.drop("group", axis=1)
    #     if count == 0:
    #         result_df = current_result
    #     else:
    #         result_df = pd.concat([result_df, current_result], axis=0, ignore_index=True)
    
    # return result_df
    result_df = None
    event_label_list = df["event_label"].unique()
    start = 'onset'
    end = 'offset'
    event_label = 'event_label'
    # for count, current_event_label in enumerate(event_label_list):
    start = df[start].values
    end = df[end].values
    event_label_id = df[event_label].values
    graph = (start <= end[:, None]) & (end >= start[:, None]) & (event_label_id == event_label_id[:, None])
    n_components, indices = connected_components(graph, directed=False)
    # if count == 0:
    result_df = df.groupby(indices).aggregate({'event_label': 'first', 'onset': 'min','offset': 'max'})
    # else:
    #     temp_df = df.groupby(indices).aggregate({'event_label': 'first', 'onset': 'min','offset': 'max'})
    #     result_df = pd.concat([result_df, temp_df], axis=0, ignore_index=True)
    return result_df
    pass

def over(df):
    result_df = None
    event_label_list = df["event_label"].unique()
    for count, current_event_label in enumerate(event_label_list):
        current_df = df.loc[df['event_label'] == current_event_label]
        current_df = current_df.sort_values(by=['onset'])
        current_df = current_df.reset_index(drop=True)
        # current_df
        min = current_df.loc[0, "onset"]
        max = current_df.loc[0, "offset"]
        for i in range(len(current_df.index)-1,0, -1):
            if (current_df.loc[i, "onset"] > min and current_df.loc[i, "offset"]<max):
                current_df = current_df.drop([i])
        if count == 0:
                result_df = current_df
        else:
            result_df = pd.concat([result_df, current_df], axis=0, ignore_index=True)
    return result_df

def ena_data_preprocess(dataset_root):
    annotation_path = os.path.join(dataset_root, "annotation")
    recording_path = os.path.join(dataset_root, "wav")
    domain_name_list = [name for name in os.listdir(annotation_path) if "Recording" in name]
    
    saved_path = os.path.join(cfg.dataset_root, "preprocess")
    
    mel_saved_path = os.path.join(saved_path, "wav")
    annotation_saved_path = os.path.join(saved_path, "annotation")
    
    if not os.path.exists(mel_saved_path):
        os.makedirs(mel_saved_path)
    
    if not os.path.exists(annotation_saved_path):
        os.makedirs(annotation_saved_path)
        
        
    # iterate through all domain and preprocess the data inside
    for domain_name in domain_name_list:
        current_annotation_path = os.path.join(annotation_path, domain_name)
        current_recording_path = os.path.join(recording_path, domain_name)
        
        audio_files_path_list = glob(os.path.join(current_recording_path, "*.wav"))
        # iterate through all wave files in the domain folder
        for current_audio_files_path in audio_files_path_list:
            # find corresponding annotation file
            # 1. get base name and eliminate extention
            wav_name = os.path.splitext(os.path.basename(current_audio_files_path))[0]
            current_annotation_file_path = glob(os.path.join(current_annotation_path, wav_name + "*.txt"))[0]
            
            audio, sr = librosa.load(current_audio_files_path, sr=cfg.sr)
            annotation_df = pd.read_csv(current_annotation_file_path, sep="\t")
            
            # eliminate the bird not in the bird list
            annotation_df.rename(columns = {'Begin Time (s)':'onset', 'End Time (s)':'offset', 'Species':'event_label'}, inplace = True)
            annotation_df = annotation_df[annotation_df["event_label"].isin(cfg.bird_list)]
            
            # cut audio data every 10 sec, puting into list
            audio_seg_list = librosa.util.frame(audio, frame_length=cfg.seg_sec*sr, hop_length=cfg.seg_sec*sr, axis=0)
            
            

            df_current = annotation_df[["onset", "offset", "event_label"]]
            for count, audio_seg in enumerate(audio_seg_list):
                # 10 sec every time
                # mel spectrogram
                mel = preprocess(audio=audio_seg, compute_log=False)
                
                # get corresponding annotation
                current_time_min = count * cfg.seg_sec
                current_time_max = (count + 1) * cfg.seg_sec
                
                # cases that cross the segment
                df_current = overlap(df = df_current, time = current_time_max)
                # df_overlap = df_current.loc[(df_current["onset"] < current_time_max) & (df_current["offset"] > current_time_max)]
                
                
                df_current_filter = df_current.loc[(df_current["onset"] >= current_time_min) & (df_current["offset"] < current_time_max)]
                df_current_filter["onset"] = df_current_filter["onset"] - current_time_min
                df_current_filter["offset"] = df_current_filter["offset"] - current_time_min
                
                # df_current_filter_2 = df_current_filter
                # same event_label but overlap
                df_current_filter_2 = same_event_label_overlap(df = df_current_filter)
                # df_current_filter_2 = over(df = df_current_filter_2)
                # df_current_filter = df_current_filter.sort_values(by=['onset'])
                if df_current_filter_2 is None:
                    df_current_filter_2 = pd.DataFrame(columns=["onset", "offset", "event_label"])
                df_current_filter_2 = df_current_filter_2.drop_duplicates()
                # save mel spectrogram
                np.save(os.path.join(mel_saved_path, wav_name + "_" + str(count)), mel)
                
                # save annotation file
                df_current_filter_2.to_csv(os.path.join(annotation_saved_path, wav_name + "_" + str(count) + ".txt"), sep="\t", index=False)
            
            # print(domain_name + " done")
    print("end")
    # pass
def data_split(dataset_root):
    
    random.seed(1215)

    saved_path = os.path.join(cfg.dataset_root, "preprocess")
    mel_saved_path = os.path.join(saved_path, "wav")
    annotation_saved_path = os.path.join(saved_path, "annotation")
    
    train_saved_path = os.path.join(cfg.dataset_root, "train_preprocess")
    train_mel_saved_path = os.path.join(train_saved_path, "wav")
    train_annotation_saved_path = os.path.join(train_saved_path, "annotation")

    if not os.path.exists(train_mel_saved_path):
        os.makedirs(train_mel_saved_path)
        os.makedirs(train_annotation_saved_path)

    val_saved_path = os.path.join(cfg.dataset_root, "val_preprocess")
    val_mel_saved_path = os.path.join(val_saved_path, "wav")
    val_annotation_saved_path = os.path.join(val_saved_path, "annotation")
    
    if not os.path.exists(val_mel_saved_path):
        os.makedirs(val_mel_saved_path)
        os.makedirs(val_annotation_saved_path)


    total_mel_file_list = glob(os.path.join(mel_saved_path, "*.npy"))
    
    train_mel_set =  set(random.sample(set(total_mel_file_list), int(len(total_mel_file_list)/2)))
    val_mel_set = set(total_mel_file_list) - train_mel_set
    
    # print(len(train_mel_set))
    for mel_file_path in train_mel_set:
        file_name = os.path.splitext(os.path.basename(mel_file_path))[0]
        annotation_file_path = glob(os.path.join(annotation_saved_path, file_name + ".txt"))[0]
        shutil.copy(mel_file_path, train_mel_saved_path)
        shutil.copy(annotation_file_path, train_annotation_saved_path)
        # print(annotation_file_path)

    for mel_file_path in val_mel_set:
        file_name = os.path.splitext(os.path.basename(mel_file_path))[0]
        annotation_file_path = glob(os.path.join(annotation_saved_path, file_name + ".txt"))[0]
        shutil.copy(mel_file_path, val_mel_saved_path)
        shutil.copy(annotation_file_path, val_annotation_saved_path)

if __name__ == '__main__':
    dataset_root = cfg.dataset_root
    ena_data_preprocess(cfg.dataset_root)
    data_split(cfg.dataset_root)
    # syn_data_preprocess()