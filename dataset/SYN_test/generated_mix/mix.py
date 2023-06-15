import librosa
import soundfile as sf
# import data_config as cfg
audio_file_1 = librosa.load('/home/fumchin/data/bsed_20/dataset/SYN_test/generated_mix/04.wav', sr=32000)
audio_file_2 = librosa.load('/home/fumchin/data/bsed_20/dataset/SYN_test/generated_mix/09.wav', sr=32000)

audio_mix_array = 0.5 * (audio_file_1[0] + audio_file_2[0])
sf.write('/home/fumchin/data/bsed_20/dataset/SYN_test/generated_mix/0.wav', audio_mix_array, samplerate=32000)