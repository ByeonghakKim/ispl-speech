import math
import numpy as np
from datasets import audio
from scipy.special import jv
from utils import noise_estimation


def spectral_subtraction(wav, sample_rate, process_type='1'):

	# STFT parameters
	win_size = int(np.round(sample_rate * 0.032))
	n_fft = 2 ** math.ceil(math.log(win_size, 2))
	hop_size = int(np.floor(win_size * .5))

	spec = audio.librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_size, win_length=win_size)

	# divide into magnitude and phase
	mag = abs(spec)
	phase = np.exp(1.j * np.angle(spec))

	if process_type is '1':
		# estimate noise variance during the first half seconds
		noise_spec = audio.librosa.stft(y=wav[:int(sample_rate / 2)], n_fft=n_fft, hop_length=hop_size, win_length=win_size)
		noise_mag = np.abs(noise_spec)
		noise_est = np.mean((noise_mag ** 2), axis=-1)
		noise_pow = np.matlib.repmat(np.reshape(noise_est, (-1, 1)), 1, mag.shape[-1])
	else:
		# estimate the variance of the noise using minimum statistics noise PSD estimation; lambda_d(k)
		noise_pow = noise_estimation.estnoisem(np.transpose(mag ** 2), hop_size / sample_rate)
		noise_pow = noise_pow.T

	pow_est = (mag ** 2 - noise_pow) * (mag ** 2 - noise_pow > 0)

	mag_est = np.sqrt(pow_est)
	spec_est = mag_est * phase
	wav_est = audio.librosa.istft(spec_est, hop_length=hop_size, win_length=win_size)

	return wav_est


def mmse_stsa(wav, sample_rate, process_type='1'):

	# algorithm parameters
	alpha = .98
	max_post_snr_db = 40
	max_post_snr = 10 ** (max_post_snr_db / 20)
	min_post_snr_db = 0
	min_post_snr = 10 ** (min_post_snr_db / 20)

	# STFT parameters
	win_size = int(np.round(sample_rate * 0.032))
	n_fft = 2 ** math.ceil(math.log(win_size, 2))
	hop_size = int(np.floor(win_size * .5))

	spec = audio.librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_size, win_length=win_size)

	mag = np.abs(spec)
	phase = np.exp(1.j * np.angle(spec))

	if process_type is '1':
		# estimate noise variance during the first half seconds
		noise_spec = audio.librosa.stft(y=wav[:int(sample_rate / 2)], n_fft=n_fft, hop_length=hop_size, win_length=win_size)
		noise_mag = np.abs(noise_spec)
		noise_est = np.mean((noise_mag ** 2), axis=-1)
		noise_pow = np.matlib.repmat(np.reshape(noise_est, (-1, 1)), 1, mag.shape[-1])
	else:
		# estimate the variance of the noise using minimum statistics noise PSD estimation; lambda_d(k)
		noise_pow = noise_estimation.estnoisem(np.transpose(mag ** 2), hop_size / sample_rate)
		noise_pow = noise_pow.T

	# MMSE
	prev_prior = 1.
	total_estimated_mag = list()
	for idx in range(mag.shape[1]):
		curr_noisy_mag = mag[:, idx]
		curr_noise_pow = noise_pow[:, idx]

		post_snr = curr_noisy_mag**2 / curr_noise_pow
		post_snr[post_snr > max_post_snr] = max_post_snr
		post_snr[post_snr < min_post_snr] = min_post_snr

		prior_snr = alpha * prev_prior + (1-alpha) * (post_snr - 1) * (post_snr - 1 > 0)

		v = prior_snr * post_snr / (1 + prior_snr)
		# for large values (~20 dB) of the instantaneous SNR, the gain function is similar to the Wiener gain.
		# instantaneous SNR: (a posteriori SNR - 1)
		gain = prior_snr / (1 + prior_snr)
		if any(v < 1):
			gain[v < 1] = (math.gamma(1.5) * np.sqrt(v[v < 1])) / post_snr[v < 1] * np.exp(-1 * v[v < 1] / 2)\
						  * ((1 + v[v < 1]) * bessel(0, v[v < 1]/2) + v[v < 1] * bessel(1, v[v < 1]/2))

		prev_prior = (gain ** 2) * post_snr
		estimated_mag = gain * curr_noisy_mag
		total_estimated_mag.append(estimated_mag)

	total_estimated_mag = np.asarray(total_estimated_mag).T
	est_spec = total_estimated_mag * phase
	est_wav = audio.librosa.istft(est_spec, hop_length=hop_size, win_length=win_size)

	return est_wav


def noise_estimate(pow_noise_spec):
	return np.mean(pow_noise_spec, axis=1)


def bessel(v, x):
	return ((1j ** (-v)) * jv(v, 1j * x)).real


def svd_enhancement(wav, sample_rate):
	from scipy.linalg import toeplitz, svd
	# algorithm parameters

	# num_frame: # of frames
	# num_sample: samples of each frame
	num_sample = 600
	num_frame = int(len(wav) / num_sample)
	l = int(num_sample / 3)
	min_singular_value = .25

	est_wav = np.zeros(num_frame * num_sample)

	for frame in range(num_frame):

		y = wav[frame * num_sample:frame * num_sample + num_sample]
		# generate toeplitz matrix
		col = y[l:]
		row = y[:l]
		row = row[::-1]
		Y = toeplitz(col, row)

		# SVD
		U, S, Vh = svd(Y)

		# in this case, p is fixed. because we didn't have noise-only wav file.
		p = len(S[S > min_singular_value])
		X = np.dot(U[:, :p], np.dot(np.diag(S[:p]), Vh[:p, :]))

		frame_enhanced = np.zeros(num_sample)
		idx = 0
		# X_hat --> toeplitz form and reconstruct enhanced signal from toeplitz matrix
		for j in range(-1 * X.shape[0] + 1, X.shape[1], 1):
			temp = X.diagonal(j)
			avg = np.mean(temp)
			frame_enhanced[idx] = avg
			idx += 1
		frame_enhanced = frame_enhanced[::-1]
		est_wav[frame * num_sample:frame * num_sample + num_sample] = frame_enhanced

	return est_wav
