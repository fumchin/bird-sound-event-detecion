import os, os.path
from pydub import AudioSegment
import data_config as cfg

if __name__ == '__main__':
    fg_folder = cfg.fg_folder
    bird_name_list = os.listdir(fg_folder)
    for bird_name in bird_name_list:
        current_folder = os.path.join(fg_folder, bird_name)
        audio_file_list = os.listdir(current_folder)
        print(audio_file_list)
        for audio_file in audio_file_list:
            name, ext = os.path.splitext(audio_file)
            audio_file_path = os.path.join(current_folder, audio_file)
            if ext == ".mp3":
                mp3_sound = AudioSegment.from_mp3(audio_file_path)
                #rename them using the old name + ".wav"
                mp3_sound.set_frame_rate(cfg.sr)
                mp3_sound.export(os.path.join(current_folder, "{0}.wav".format(name)), format="wav")
                os.remove(audio_file_path)