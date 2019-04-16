from datasets import audio
from utils import speech_enhancement
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa.display


def main():
    process_type = '1'

    input_path = './sample.wav'
    output_path = input_path.split('.wav')[0] + '_enhanced_mmse' + process_type + '.wav'
    wav, sample_rate = audio.load_wav(input_path, sr=None)
    wav = wav / np.abs(wav).max() * 0.999

    clean_wav = speech_enhancement.mmse_stsa(wav, sample_rate=sample_rate, process_type=process_type)

    # Plot result
    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(wav, sr=sample_rate)
    plt.title('Noisy Time Signal')
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(clean_wav, sr=sample_rate)
    plt.title('Estimated Clean Time Signal')

    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(
        librosa.power_to_db(librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_fft=1024, hop_length=512),
                            ref=np.max), sr=sample_rate, x_axis='time', y_axis='linear')
    plt.title('Noisy Spectrogram')
    plt.colorbar(format='%+2.0f dB', boundaries=np.linspace(-70, 0, 10))
    plt.subplot(2, 1, 2)
    librosa.display.specshow(
        librosa.power_to_db(librosa.feature.melspectrogram(y=clean_wav, sr=sample_rate, n_fft=1024, hop_length=512),
                            ref=np.max), sr=sample_rate, x_axis='time', y_axis='linear')
    plt.title('Estimated Clean Spectrogram')
    plt.colorbar(format='%+2.0f dB', boundaries=np.linspace(-70, 0, 10))
    plt.tight_layout()
    plt.show()

    clean_wav *= 32767 / max(0.01, np.max(np.abs(clean_wav)))
    # proposed by @dsmiller
    wavfile.write(output_path, sample_rate, clean_wav.astype(np.int16))


if __name__ == '__main__':
    main()
