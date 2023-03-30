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
# from statsmodels.stats.weightstats import ztest
# import seaborn as snsW


def visualization(ADA_synth_feature, ADA_real_feature, ADA_path=None, No_ADA_path=None, sample_num=1000):
    print("start visualization")

    ADA_tsne_path = os.path.join(ADA_path, 'tsne(frame).npy')

    # if not os.path.exists(ADA_tsne_path):
    synth_sample_index = np.random.choice(ADA_synth_feature.shape[0], sample_num)
    real_sample_index = np.random.choice(ADA_real_feature.shape[0], sample_num)
    # ADA_synth_feature_sample = ADA_synth_feature[synth_sample_index]
    # ADA_real_feature_sample = ADA_real_feature[real_sample_index]
    # # ADA_synth_label_sample = ADA_synth_label[synth_sample_index]
    # ADA_real_label_sample = ADA_real_label[real_sample_index]
    ADA_synth_feature_sample = ADA_synth_feature
    ADA_real_feature_sample = ADA_real_feature
    # ADA_synth_label_sample = ADA_synth_label
    # ADA_real_label_sample = ADA_real_label

    # ADA_synth_feature_sample = ADA_synth_feature_sample.reshape(ADA_synth_feature_sample.shape[0]*ADA_synth_feature_sample.shape[1], ADA_synth_feature_sample.shape[2])
    # ADA_real_feature_sample = ADA_real_feature_sample.reshape(ADA_real_feature_sample.shape[0]*ADA_real_feature_sample.shape[1], ADA_real_feature_sample.shape[2])
    # ADA_synth_label_sample = ADA_synth_label_sample.reshape(ADA_synth_label_sample.shape[0]*ADA_synth_label_sample.shape[1], ADA_synth_label_sample.shape[2])
    # ADA_real_label_sample = ADA_real_label_sample.reshape(ADA_real_label_sample.shape[0]*ADA_real_label_sample.shape[1], ADA_real_label_sample.shape[2])

    ADA_feature = np.concatenate((ADA_synth_feature_sample, ADA_real_feature_sample), axis=0)
    # print(ADA_feature.shape)
    # ADA_label = np.concatenate((ADA_synth_label_sample, ADA_real_label_sample), axis=0)
    # ADA_label = [i[0] for i in ADA_label]
    # ADA_feature = scaler.fit_transform(ADA_feature)
    ADA_feature = TSNE(n_components=2).fit_transform(ADA_feature)
    # ADA_feature = FastICA(n_components=2).fit_transform(ADA_feature)
    # pca = PCA(n_components=2)
    # ADA_feature = pca.fit_transform(ADA_feature)
    # ADA_recon = pca.inverse_transform(ADA_feature)
    np.save(ADA_tsne_path, ADA_feature)
    # print(ADA_recon.shape)
    ADA_feature = np.load(ADA_tsne_path)

    ADA_synth_feature_sample = ADA_feature[:int(ADA_synth_feature_sample.shape[0])]
    ADA_real_feature_sample = ADA_feature[int(ADA_synth_feature_sample.shape[0]):]
    ADA_synth_label = ['synth'] * int(ADA_synth_feature_sample.shape[0])
    ADA_real_label = ['real'] * int(ADA_real_feature_sample.shape[0])
    ADA_label = np.array(ADA_synth_label + ADA_real_label)
    s_score = silhouette_score(ADA_feature, ADA_label)
    print('silhouette score: ', s_score)
    # ADA_synth_label_sample = ADA_label[:int(ADA_feature.shape[0]/2)]
    # ADA_real_label_sample = ADA_label[int(ADA_feature.shape[0]/2):]
    # df_real = pd.DataFrame()
    # df_real['x'] = ADA_real_feature_sample[:,0].tolist()
    # df_real['y'] = ADA_real_feature_sample[:,1].tolist()
    # df_real['label'] = ADA_real_label_sample

    # df['type'] = 0
    # df['type'][:int(ADA_feature.shape[0]/2)] = 'real'
    # df['type'][int(ADA_feature.shape[0]/2):] = 'synth'
    plt.figure()
    # plt.scatter(ADA_synth_feature_sample[:,0], ADA_synth_feature_sample[:,1], alpha=0.6, s=10, marker='x', label='synth')
    #plt.scatter(df['x'], df['y'], alpha=0.6, s=10, c=df['label'], label='synth')
    # sns.scatterplot(x="x", y="y", hue="label", palette="deep", linewidth=0, data=df_real, s=10)
    plt.scatter(ADA_synth_feature_sample[:,0], ADA_synth_feature_sample[:,1], alpha=0.6, s=10, marker='x', label='synth')
    plt.scatter(ADA_real_feature_sample[:,0], ADA_real_feature_sample[:,1], alpha=0.3, s=10, c='r', label='real')
    # transfer back to audio
    # print(ADA_synth_feature_sample.shape)
    # y_synth_mel = ADA_recon[:int(ADA_synth_feature_sample.shape[0]),:]
    # y_real_mel = ADA_recon[int(ADA_synth_feature_sample.shape[0]):,:]
    # print(y_synth_mel.shape)
    # print(y_real_mel.shape)

    # y_synth = librosa.feature.inverse.mel_to_audio(y_synth_mel, sr=16000, n_fft=2048, hop_length=255)
    # y_real = librosa.feature.inverse.mel_to_audio(y_real_mel, sr=16000, n_fft=2048, hop_length=255)
    # print('hi')
    # write(os.path.join(ADA_path, 'synth_PCA_recon.wav'), 16000, y_synth)
    # write(os.path.join(ADA_path, 'real_PCA_recon.wav'), 16000, y_real)
    # plt.scatter(ADA_real_feature_sample[:,0], ADA_real_feature_sample[:,1], alpha=0.3, s=10, c='r', label='real')
    plt.legend()
    # plt.title('syn vs real')
    # plt.xlim(-60, 80)
    # plt.ylim(-90, 90)
    plt.axis('off')
    plt.savefig(os.path.join(ADA_path, 'syn vs real(frame).png'))



def svm_classfication(ADA_synth_feature, ADA_real_feature, No_ADA_synth_feature, No_ADA_real_feature, sample_num=1000):
    print("start doing classification")
    synth_sample_index = np.random.choice(ADA_synth_feature.shape[0], sample_num)
    real_sample_index = np.random.choice(ADA_real_feature.shape[0], sample_num)
    ADA_synth_feature_sample = ADA_synth_feature[synth_sample_index]
    ADA_real_feature_sample = ADA_real_feature[real_sample_index]
    ADA_feature = np.concatenate((ADA_synth_feature_sample, ADA_real_feature_sample), axis=0)

    No_ADA_synth_feature_sample = No_ADA_synth_feature[synth_sample_index]
    No_ADA_real_feature_sample = No_ADA_real_feature[real_sample_index]
    No_ADA_feature = np.concatenate((No_ADA_synth_feature_sample, No_ADA_real_feature_sample), axis=0)
    target = np.concatenate((np.ones(len(No_ADA_synth_feature_sample)), np.zeros(len(No_ADA_real_feature_sample))), axis=0)

    # use cross-validation
    classifier = SVC()
    ADA_result = cross_val_score(classifier, ADA_feature, target, cv=5)
    No_ADA_result = cross_val_score(classifier, No_ADA_feature, target, cv=5)
    print("ADA score is {}".format(sum(ADA_result)/len(ADA_result)))
    print("No ADA score is {}".format(sum(No_ADA_result)/len(No_ADA_result)))


if __name__ == '__main__':
    # model_list = ['fum_IP_fusion', 'fum_IP_fusion_2', 'fum_IP_fusion_3', 'fum_IP_fusion_4', 'fum_IP_fusion_5', 'fum_IP_fusion_6']
    # model_list = ['CRNN_0323_fpn_ada_official_default_lr_advw_10', 'CRNN_0323_fpn_ada_official_default_lr_advw_5', 'CRNN_fpn']
    # model_list = ['only_fixed_ada_no_prediction_', 'only_ada_no_prediction', 'only_ada_no_prediction_2']
    model_list = ['CRNN_fpn_3000_cdan_test']
    for model_name in model_list:
        print(model_name)
        ADA_path = os.path.join("./stored_data", model_name, "embedded_features")
        # ADA_path = "./DA/embedded_feature/fum_IP_2"# embedded feature dir using domain adaptation
        ADA_synth_path = os.path.join(ADA_path, "syn")
        ADA_real_path = os.path.join(ADA_path, "train")
        # ADA_real_path = os.path.join(ADA_path, "strong")

        # ADA_synth_label_path = ADA_synth_path + "_label"
        # ADA_real_label_path = ADA_real_path + "_label"

        No_ADA_path = "./DA/embedded_feature/fum_full" # embedded feature dir without using domain adaptation
        No_ADA_synth_path =  os.path.join(No_ADA_path, "syn")
        No_ADA_real_path =  os.path.join(No_ADA_path, "strong")

        # standardize scaler
        scaler = StandardScaler()
        k = os.listdir(ADA_synth_path)
        # load and concatenate ADA synthetic features
        for feature_path in os.listdir(ADA_synth_path):
            if feature_path == os.listdir(ADA_synth_path)[0]:
                ADA_synth_feature = np.load(os.path.join(ADA_synth_path, feature_path))
            else:
                ADA_synth_feature = np.concatenate((ADA_synth_feature, np.load(os.path.join(ADA_synth_path, feature_path))), axis=0)
        ADA_synth_feature = ADA_synth_feature.reshape(ADA_synth_feature.shape[0], ADA_synth_feature.shape[1]*ADA_synth_feature.shape[2])

        # for feature_label_path in os.listdir(ADA_synth_label_path):
        #     if feature_label_path == os.listdir(ADA_synth_label_path)[0]:
        #         ADA_synth_label = np.load(os.path.join(ADA_synth_label_path, feature_label_path))
        #     else:
        #         ADA_synth_label = np.concatenate((ADA_synth_label, np.load(os.path.join(ADA_synth_label_path, feature_label_path))), axis=0)
        # # ADA_synth_label = ADA_synth_label.reshape(ADA_synth_label.shape[0]*ADA_synth_label.shape[1], ADA_synth_label.shape[2])
        # assert ADA_synth_label.shape[0] == ADA_synth_feature.shape[0]

        # load and concatenate ADA real features
        for feature_path in os.listdir(ADA_real_path):
            if feature_path == os.listdir(ADA_real_path)[0]:
                ADA_real_feature = np.load(os.path.join(ADA_real_path, feature_path))
            else:
                ADA_real_feature = np.concatenate((ADA_real_feature, np.load(os.path.join(ADA_real_path, feature_path))), axis=0)
        ADA_real_feature = ADA_real_feature.reshape(ADA_real_feature.shape[0], ADA_real_feature.shape[1]* ADA_real_feature.shape[2])

        # for feature_label_path in os.listdir(ADA_real_label_path):
        #     if feature_label_path == os.listdir(ADA_real_label_path)[0]:
        #         ADA_real_label = np.load(os.path.join(ADA_real_label_path, feature_label_path))
        #     else:
        #         ADA_real_label = np.concatenate((ADA_real_label, np.load(os.path.join(ADA_real_label_path, feature_label_path))), axis=0)
        # # ADA_real_label = ADA_real_label.reshape(ADA_real_label.shape[0]*ADA_real_label.shape[1], ADA_real_label.shape[2])
        # assert ADA_real_label.shape[0] == ADA_real_feature.shape[0]

        # load and concatenate none ADA synthetic features
        # for feature_path in os.listdir(No_ADA_synth_path):
        #     if feature_path == os.listdir(No_ADA_synth_path)[0]:
        #         No_ADA_synth_feature = np.load(os.path.join(No_ADA_synth_path, feature_path))
        #     else:
        #         No_ADA_synth_feature = np.concatenate((No_ADA_synth_feature, np.load(os.path.join(No_ADA_synth_path, feature_path))), axis=0)
        # No_ADA_synth_feature = No_ADA_synth_feature.reshape(No_ADA_synth_feature.shape[0]*No_ADA_synth_feature.shape[1], No_ADA_synth_feature.shape[2])

        # # load and concatenate none ADA real features
        # for feature_path in os.listdir(No_ADA_real_path):
        #     if feature_path == os.listdir(ADA_real_path)[0]:
        #         No_ADA_real_feature = np.load(os.path.join(No_ADA_real_path, feature_path))
        #     else:
        #         No_ADA_real_feature = np.concatenate((No_ADA_real_feature, np.load(os.path.join(No_ADA_real_path, feature_path))), axis=0)
        # No_ADA_real_feature = No_ADA_real_feature.reshape(No_ADA_real_feature.shape[0]*No_ADA_real_feature.shape[1], No_ADA_real_feature.shape[2])

        print('finish loading')


        # visualize with tsne
        start = time.time()
        # visualization(ADA_synth_feature, ADA_real_feature, No_ADA_synth_feature, No_ADA_real_feature, ADA_path=ADA_path, No_ADA_path=No_ADA_path)
        visualization(ADA_synth_feature, ADA_real_feature, ADA_path=ADA_path, No_ADA_path=No_ADA_path)
        end = time.time()
        print('finish visualization takes {} s'.format(end-start))


        # svm classification
        # start = time.time()
        # svm_classfication(ADA_synth_feature, ADA_real_feature, No_ADA_synth_feature, No_ADA_real_feature)
        # end = time.time()
        # print('finish svm classification takes {} s'.format(end-start))

