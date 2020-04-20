import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Audio
	num_mels=80, # Number of mel-spectrogram channels
	w=19,
	u=9,
	rescale=True, # Whether to rescale audio prior to preprocessing
	rescaling_max=0.999, # Rescaling value
	trim_silence=False, # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	clip_mels_length=True, # For cases of OOM (Not really recommended, working on a workaround)

	# Mel spectrogram
	n_fft=1024, # Extra window size is filled with 0 paddings to match this parameter
	hop_size=160, # For 22050Hz, 275 ~= 12.5 ms
	win_size=400, # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft)
	sample_rate=16000, # 22050 Hz (corresponding to ljspeech dataset)
	frame_shift_ms=None,

	# Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization=True,
	allow_clipping_in_normalization=True, # Only relevant if mel_normalization = True
	symmetric_mels=False, # Whether to scale the data to be symmetric around 0
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

	# Limits
	min_level_db=-100,
	ref_level_db=20,
	fmin=0, # Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax=8000,

	###########################################################################################################################################

	# VAD
	hidden_dim=512, # dimension of hidden layer
	lstm_cell_size=256,
	lstm_num_layers=3,
	num_glimpses=7,
	glimpse_hidden=128,
	action_hidden=256,

	###########################################################################################################################################

	# Proposed
	layers=4,
	conv_channels=16,
	filter_width=3,
	lstm_units=256,
	num_proj=128,
	prenet_units=256,
	num_heads=4,
	num_att_units=128,


	###########################################################################################################################################


	# VAD Training
	vad_random_seed=777, #Determines initial graph and operations (i.e: model) random state for reproducibility
	vad_swap_with_cpu=False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

	vad_batch_size=512, #number of training samples on each training steps
	vad_reg_weight=1e-6, #regularization weight (for L2 regularization)
	vad_scale_regularization=False, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)

	vad_test_size=0.05, #% of data to keep as test data, if None, vad_test_batches must be not None
	vad_test_batches=None, #number of test batches (For Ljspeech: 10% ~= 41 batches of 32 samples)
	vad_data_random_state=777, #random state for train test split repeatability

	vad_decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
	vad_start_decay=50000, #Step at which learning decay starts
	vad_decay_steps=25000, #Determines the learning rate decay slope (UNDER TEST)
	vad_decay_rate=0.6, #learning rate decay   (UNDER TEST)
	vad_initial_learning_rate=1e-3, #starting learning rate
	vad_final_learning_rate=1e-5, #minimal learning rate

	vad_adam_beta1=0.9, #AdamOptimizer beta1 parameter
	vad_adam_beta2=0.999, #AdamOptimizer beta2 parameter
	vad_adam_epsilon=1e-6, #AdamOptimizer Epsilon parameter

	vad_zoneout_rate=0.1, #zoneout rate for all LSTM cells in the network
	vad_dropout_rate=0.5, #dropout rate for all convolutional layers + prenet
	vad_moving_average_decay=.99,

	vad_clip_gradients=True, #whether to clip gradients
	)


def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
	return 'Hyperparameters:\n' + '\n'.join(hp)
