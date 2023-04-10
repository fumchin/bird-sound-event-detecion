import pandas as pd
import os, os.path
import config as cfg
from glob import glob
import librosa
import soundfile as sf
# from pydub import AudioSegment

if __name__ == "__main__":
    review_saved_path = os.path.join("/home/fumchin/data/bsed_20/dataset", "review")
    bird_list = cfg.bird_list
    for bird in bird_list:
        current_review_saved_path = os.path.join(review_saved_path, bird)
        if not os.path.exists(current_review_saved_path):
            os.makedirs(current_review_saved_path)


    dataset_root = cfg.dataset_root
    annotation_path = os.path.join(dataset_root, "annotation")
    recording_path = os.path.join(dataset_root, "wav")
    domain_name_list = [name for name in os.listdir(annotation_path) if "Recording" in name]



    for domain_name in domain_name_list:
        current_annotation_path = os.path.join(annotation_path, domain_name)
        current_recording_path = os.path.join(recording_path, domain_name)
        
        audio_files_path_list = glob(os.path.join(current_recording_path, "*.wav"))
        # iterate through all wave files in the domain folder
        for current_audio_files_path in audio_files_path_list:
            wav_name = os.path.splitext(os.path.basename(current_audio_files_path))[0]
            current_annotation_file_path = glob(os.path.join(current_annotation_path, wav_name + "*.txt"))[0]
            annotation_df = pd.read_csv(current_annotation_file_path, sep="\t")

            audio, sr = librosa.load(current_audio_files_path, sr=cfg.sr)
            for index, row in annotation_df.iterrows():
                onset_index = int(row['onset'] * sr)
                offset_index = int(row['offset'] * sr)
                bird_name = row['event_label']

                current_audio = audio[onset_index:offset_index]
                saved_path = os.path.join(review_saved_path, bird_name)
                file_num = len(glob(os.path.join(saved_path, '*.wav')))
                try:
                    sf.write(os.path.join(saved_path,str(file_num)+".wav"), current_audio, sr)
                except:
                    pass


    
    



