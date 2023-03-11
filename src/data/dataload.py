import bisect
import os
import os.path
import numpy as np
import pandas as pd
from glob import glob
import torch
import data.config as cfg
from data.Transforms import get_transforms
# import config as cfg
# from Transforms import get_transforms

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class ENA_Dataset(Dataset):
    def __init__(self, preprocess_dir, encod_func, transform, compute_log=False):
        self.sample_rate = cfg.sr
        self.preprocess_dir = preprocess_dir
        # self.encode_function = encode_function
        self.pooling_time_ratio = cfg.pooling_time_ratio
        self.n_frames = cfg.max_frames // self.pooling_time_ratio
        self.hop_size = cfg.hop_size
        self.transform = transform
        self.encod_func = encod_func
        
        self.annotation_dir = os.path.join(self.preprocess_dir, "annotation")
        self.feature_dir = os.path.join(self.preprocess_dir, "wav")
        self.feature_file_list = glob(os.path.join(self.feature_dir, "*.npy"))
        self.labels = cfg.bird_list
        
    def __len__(self):
        return len(self.feature_file_list)
        
    def __getitem__(self, index):
        # get selected file
        selected_file_path = self.feature_file_list[index]
        features = np.load(selected_file_path)
        
        # get its annotation
        feature_file_name = os.path.splitext(os.path.basename(selected_file_path))[0]
        annotation_file_path = glob(os.path.join(self.annotation_dir, feature_file_name + ".txt"))[0]
        # read with pandas
        df = pd.read_csv(annotation_file_path, sep="\t")
        # target = self.encode(df)
        if self.encod_func is not None:
            target = self.encod_func(df)
        else:
            target = self.encode(df)
            print("no encoded function")

        if self.transform is not None:
            sample = self.transform((features, target))
        else:
            sample = (features, target)
        return (sample, selected_file_path)
        
        
        pass
    
    def encode(self, label_df):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        y = np.zeros((self.n_frames, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            # if {"onset", "offset", "event_label"}.issubset(label_df.columns):
            for _, row in label_df.iterrows():
                i = self.labels.index(row["Species"])
                onset = int(row["Begin Time (s)"] * self.sample_rate // self.hop_size // self.pooling_time_ratio)
                offset = int(row["End Time (s)"] * self.sample_rate // self.hop_size // self.pooling_time_ratio)
                y[onset:offset, i] = 1  # means offset not included (hypothesis of overlapping frames, so ok)
        return y
    
class SYN_Dataset(Dataset):
    def __init__(self, preprocess_dir, encod_func, transform, compute_log=False):
        self.sample_rate = cfg.sr
        self.preprocess_dir = preprocess_dir
        # self.encode_function = encode_function
        self.pooling_time_ratio = cfg.pooling_time_ratio
        self.n_frames = cfg.max_frames // self.pooling_time_ratio
        self.hop_size = cfg.hop_size
        self.transform = transform
        self.encod_func = encod_func
        
        self.annotation_dir = os.path.join(self.preprocess_dir, "annotation")
        self.feature_dir = os.path.join(self.preprocess_dir, "wav")
        self.feature_file_list = glob(os.path.join(self.feature_dir, "*.npy"))
        self.labels = cfg.bird_list
        
    def __len__(self):
        return len(self.feature_file_list)
        
    def __getitem__(self, index):
        # get selected file
        selected_file_path = self.feature_file_list[index]
        features = np.load(selected_file_path)
        
        # get its annotation
        feature_file_name = os.path.splitext(os.path.basename(selected_file_path))[0]
        annotation_file_path = glob(os.path.join(self.annotation_dir, feature_file_name + ".txt"))[0]
        # read with pandas
        df = pd.read_csv(annotation_file_path, sep="\t")
        # target = self.encode(df)
        df.rename(
            columns={"onset": "Begin Time (s)", "offset": "End Time (s)", "event_label": "Species"},
            inplace=True,
        )
        if self.encod_func is not None:
            target = self.encod_func(df)
        else:
            target = self.encode(df)
            print("no encoded function")

        if self.transform is not None:
            sample = self.transform((features, target))
        else:
            sample = (features, target)
        return (sample, selected_file_path)
        
        
        pass
    
    def encode(self, label_df):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        y = np.zeros((self.n_frames, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            # if {"onset", "offset", "event_label"}.issubset(label_df.columns):
            for _, row in label_df.iterrows():
                i = self.labels.index(row["event_label"])
                onset = int(row["onset"] * self.sample_rate // self.hop_size // self.pooling_time_ratio)
                offset = int(row["offset"] * self.sample_rate // self.hop_size // self.pooling_time_ratio)
                y[onset:offset, i] = 1  # means offset not included (hypothesis of overlapping frames, so ok)
        return y

class ConcatDataset(Dataset):
    """
    DESED to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Args:
        datasets : sequence, list of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @property
    def cluster_indices(self):
        cluster_ind = []
        prec = 0
        for size in self.cumulative_sizes:
            cluster_ind.append(range(prec, size))
            prec = size
        return cluster_ind

    def __init__(self, datasets):
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    @property
    def df(self):
        df = self.datasets[0].df
        for dataset in self.datasets[1:]:
            df = pd.concat([df, dataset.df], axis=0, ignore_index=True, sort=False)
        return df


if __name__ == "__main__":
    # file_name = "Fuji_train_list.txt"
    # root = "/home/fumchin/data/cv/final/dataset"
    n_channel = 1
    add_axis_conv = 0
    transforms = get_transforms(cfg.max_frames, None, add_axis_conv,
                                noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    # ENA = ENA_Dataset(preprocess_dir="../../dataset/ENA/preprocess", transform=transforms, compute_log=False)
    # many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=cfg.max_frames // cfg.pooling_time_ratio)
    # encod_func = many_hot_encoder.encode_strong_df
    ENA = ENA_Dataset(preprocess_dir=cfg.feature_dir, encod_func=None, transform=None, compute_log=False)
    train_dataloader = DataLoader(ENA, batch_size=6, shuffle=True)
    a, b = next(iter(train_dataloader))
    print(a, b)
    print('f')