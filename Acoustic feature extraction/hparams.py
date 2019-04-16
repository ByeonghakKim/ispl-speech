import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Audio
	num_mels=80, # Number of mel-spectrogram channels
	rescale=True, # Whether to rescale audio prior to preprocessing
	rescaling_max=0.999, # Rescaling value
	trim_silence=False, # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	clip_mels_length=True, # For cases of OOM (Not really recommended, working on a workaround)

	# Mel spectrogram
	n_fft=2048, # Extra window size is filled with 0 paddings to match this parameter
	hop_size=200, # For 22050Hz, 275 ~= 12.5 ms
	win_size=800, # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft)
	sample_rate=16000, # 22050 Hz (corresponding to ljspeech dataset)
	frame_shift_ms=None,

	# Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization=True,
	allow_clipping_in_normalization=True, # Only relevant if mel_normalization = True
	symmetric_mels=True, # Whether to scale the data to be symmetric around 0
	max_abs_value=4., # max absolute value of data. If symmetric, data will be [-max, max] else [0, max]

	# Contribution by @begeekmyfriend
	# Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
	preemphasize=True, # whether to apply filter
	preemphasis=0.97, # filter coefficient.

	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It's preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False,
	silence_threshold=2,  # silence threshold used for sound trimming for wavenet preprocessing

	#Limits
	min_level_db=-100,
	ref_level_db=20,
	fmin=55, # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax=7600,
	)


def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
	return 'Hyperparameters:\n' + '\n'.join(hp)
