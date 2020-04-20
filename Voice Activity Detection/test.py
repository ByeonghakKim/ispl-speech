import os

import infolog
import tensorflow as tf
import numpy as np
from hparams import hparams
from datasets import audio
from modules.models import create_model
from modules.utils import data_transform
from glob import glob

log = infolog.log


def find_files(directory, pattern='**/*.WAV'):
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


def model_test_mode(model_name, hparams, global_step):
    with tf.variable_scope('VAD_model', reuse=tf.AUTO_REUSE) as scope:
        model = create_model(model_name, hparams)
        # Create Placeholders
        with tf.device('/cpu:0'):
            inputs = tf.placeholder(tf.float32, shape=(None, int(2 * (hparams.w - 1) / hparams.u + 3), hparams.num_mels))
            targets = tf.placeholder(tf.float32, shape=(None, int(2 * (hparams.w - 1) / hparams.u + 3)))
        model.initialize(inputs, targets, global_step=global_step, is_training=False, is_evaluating=True)
        return model


def main():
    input_path = './data_test/[AURORA]SA1_add_sil_SNR(-5)_airport.WAV'
    output_path = input_path.split('.WAV')[0] + '.jpg'
    load_dir = os.path.join('logs-Proposed', 'vad_pretrained')
    checkpoint_path = os.path.join(load_dir, 'vad_model.ckpt')

    print('Checkpoint path: {}'.format(checkpoint_path))
    print('Loading test data from: {}'.format(input_path))
    print('Using model: {}'.format('Proposed'))

    # Load the audio as numpy array
    wav = audio.load_wav(input_path, sr=hparams.sample_rate)
    # [-1, 1]
    wav = wav / np.abs(wav).max() * 0.999

    out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(out_dtype)
    mel_spectrogram = np.divide((mel_spectrogram - mel_spectrogram.min(axis=0)),
                                (mel_spectrogram.max(axis=0) - mel_spectrogram.min(axis=0)))
    mel_input = data_transform(mel_spectrogram.T, hparams.w, hparams.u, mel_spectrogram.min())[hparams.w:-hparams.w, :, :]
    mel_frames = len(mel_input)

    # Ensure time resolution adjustement between audio and mel-spectrogram
    pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams))

    # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
    wav = np.pad(wav, (0, pad), mode='reflect')
    assert len(wav) >= mel_frames * audio.get_hop_size(hparams)

    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    wav = wav[:mel_frames * audio.get_hop_size(hparams)]
    assert len(wav) % audio.get_hop_size(hparams) == 0

    time_steps = len(wav)

    # Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    eval_model = model_test_mode('Proposed', hparams, global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        checkpoint_state = tf.train.get_checkpoint_state(load_dir)
        print('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path))
        saver.restore(sess, checkpoint_state.model_checkpoint_path)

        vad_prediction = sess.run(eval_model.soft_prediction, feed_dict={eval_model.inputs: mel_input})[:, 0].round()
        vad_prediction_time = np.zeros_like(wav)
        time_idx = 0
        for mel_idx in range(mel_frames):
            if vad_prediction[mel_idx] == 1:
                vad_prediction_time[(time_steps//mel_frames)*mel_idx:(time_steps//mel_frames)*(mel_idx+1)] = 1
            else:
                vad_prediction_time[(time_steps//mel_frames)*mel_idx:(time_steps//mel_frames)*(mel_idx+1)] = 0
            time_idx += 1

        import matplotlib.pyplot as plt
        plt.plot(wav)
        plt.plot(vad_prediction_time)
        plt.savefig(output_path, dpi=300)
        plt.show()
        print('Test done')


if __name__ == '__main__':
    main()
