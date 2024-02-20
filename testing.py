# -------------------------------This is a Live Testing _____________________________________

import pickle  # to save model after training
import wave
from array import array
from struct import pack
from sys import byteorder
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


import librosa
import numpy as np
import pyaudio
import soundfile  # to read audio file
import speech_recognition as sr

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    
    with soundfile.SoundFile(file_name) as sound_file:
        X, sample_rate = sound_file.read(dtype="float32"), sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel_spec = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            mel = np.mean(librosa.power_to_db(mel_spec).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result


# from utils import extract_feature

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_silent(snd_data):

    return max(snd_data) < THRESHOLD


def normalize(snd_data):

    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r



def trim(snd_data):

    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):

    r = array('h', [0 for i in range(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * RATE))])
    return r


def record():

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True


        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):

    sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()



loaded_model = pickle.load(open("result/mlp_classifier.model", 'rb'))

print("Please talk")
filename = "test.wav"
a = filename

#record the file (start talking)
record_to_file(filename)

# ------------------------sound wave ______________

# --------------------------------------------Speech to text------------------------------------------

mic = sr.Recognizer()
# open the file
with sr.AudioFile(filename) as source:
    # listen for the data (load audio to memory)
    audio_data = mic.record(source)
    #extract features and reshape it
    features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = loaded_model.predict(features)[0]
    # show the result !
    Emotion= ("PREDICTED Emotion is : ", result)
    print(Emotion)
    # recognize (convert from speech to text)
    text = mic.recognize_google(audio_data)
    print(text)

# ----------------------------------------------------------------------------

rate, data = wav.read('test.wav')

plt.plot(data)
plt.show()
plt.savefig('wave.png') #to save the ploting figure
