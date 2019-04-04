import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob

import numpy as np
import scipy.io as sio
from datasets import audio


def find_files(directory, pattern='**/*.WAV'):
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


def build_from_path(hparams, input_dirs, mel_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		files = find_files(os.path.join(input_dir))
		for wav_path in files:
			futures.append(executor.submit(partial(_process_utterance, mel_dir, index, wav_path, hparams)))
			index += 1

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(mel_dir, index, wav_path, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
		return None

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	#[-1, 1]
	out = wav
	constant_values = 0.
	out_dtype = np.float32

	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	# Ensure time resolution adjustement between audio and mel-spectrogram
	pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams))

	# Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
	out = np.pad(out, (0, pad), mode='reflect')
	assert len(out) >= mel_frames * audio.get_hop_size(hparams)

	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	out = out[:mel_frames * audio.get_hop_size(hparams)]
	assert len(out) % audio.get_hop_size(hparams) == 0
	time_steps = len(out)


	# Write the spectrogram and audio to disk
	mel_filename = 'mel-{}.npy'.format(index)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example
	return (wav_path, mel_filename, time_steps, mel_frames)


def build_from_path_kaist(hparams, input_dirs, mel_dir, label_dir, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	futures = []
	index = 1
	for input_dir in input_dirs:
		files = find_files(os.path.join(input_dir))
		for wav_path in files:
			text_path = os.path.splitext(wav_path)[0] + '.mat'
			label = np.reshape(sio.loadmat(text_path)['y_label'], (-1))
			try:
				# Load the audio as numpy array
				wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
			except FileNotFoundError:  # catch missing wav exception
				print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
				return None

			# rescale wav
			if hparams.rescale:
				wav = wav / np.abs(wav).max() * hparams.rescaling_max

			# M-AILABS extra silence specific
			if hparams.trim_silence:
				wav = audio.trim_silence(wav, hparams)

			# [-1, 1]
			out = wav
			constant_values = 0.
			out_dtype = np.float32

			# Compute the mel scale spectrogram from the wav
			mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
			mel_frames = mel_spectrogram.shape[1]

			# Ensure time resolution adjustement between audio and mel-spectrogram
			pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams))

			# Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
			out = np.pad(out, (0, pad), mode='reflect')
			label = np.pad(label, (0, pad), mode='reflect')
			assert len(out) >= mel_frames * audio.get_hop_size(hparams)

			# time resolution adjustement
			# ensure length of raw audio is multiple of hop size so that we can use
			# transposed convolution to upsample
			out = out[:mel_frames * audio.get_hop_size(hparams)]
			label = label[:mel_frames * audio.get_hop_size(hparams)]
			assert len(out) % audio.get_hop_size(hparams) == 0
			label = label[::audio.get_hop_size(hparams)]

			time_steps = len(out)

			# Write the spectrogram and audio to disk
			mel_filename = 'mel-{}.npy'.format(index)
			label_filename = 'label-{}.npy'.format(index)
			np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
			np.save(os.path.join(label_dir, label_filename), label, allow_pickle=False)
			futures.append((wav_path, mel_filename, time_steps, mel_frames, label_filename))
			index += 1

	return [future for future in tqdm(futures)]
