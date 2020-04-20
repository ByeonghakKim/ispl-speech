import argparse
import os
from time import sleep

import infolog
import tensorflow as tf
from hparams import hparams
from infolog import log
from modules.train import vad_train

log = infolog.log

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def save_seq(file, sequence, input_path):
	'''Save VAD training state to disk. (To skip for future runs)
	'''
	sequence = [str(int(s)) for s in sequence] + [input_path]
	with open(file, 'w') as f:
		f.write('|'.join(sequence))


def read_seq(file):
	'''Load VAD training state from disk. (To skip if not first run)
	'''
	if os.path.isfile(file):
		with open(file, 'r') as f:
			sequence = f.read().split('|')
		return [bool(int(s)) for s in sequence[:-1]], sequence[-1]
	else:
		return [0], ''


def prepare_run(args):
	modified_hp = hparams.parse(args.hparams)
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
	run_name = args.name or args.model
	log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
	os.makedirs(log_dir, exist_ok=True)
	infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name, args.slack_url)
	return log_dir, modified_hp


def train(args, log_dir, hparams):
	state_file = os.path.join(log_dir, 'state_log')
	# Get training states
	vad_state, input_path = read_seq(state_file)

	log('\n#############################################################\n')
	log('VAD Train\n')
	log('###########################################################\n')
	checkpoint = vad_train(args, log_dir, hparams)
	tf.reset_default_graph()
	#Sleep 1/2 second to let previous graph close
	sleep(0.5)
	if checkpoint is None:
		raise ValueError('Error occured while training VAD, Exiting!')

	vad_state = 1

	if vad_state:
		log('TRAINING IS ALREADY COMPLETE!!')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--vad_input', default='training_data/train.txt')
	parser.add_argument('--name', help='Name of logging directory.')
	parser.add_argument('--model', default='Proposed')
	parser.add_argument('--input_dir', default='training_data', help='folder to contain inputs sentences/targets')
	parser.add_argument('--output_dir', default='output', help='folder to contain prediction')
	parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
	parser.add_argument('--summary_interval', type=int, default=5000, help='Steps between running summary ops')
	parser.add_argument('--checkpoint_interval', type=int, default=25000, help='Steps between writing checkpoints')
	parser.add_argument('--eval_interval', type=int, default=25000, help='Steps between eval on test data')
	parser.add_argument('--vad_train_steps', type=int, default=500000, help='total number of vad training steps')
	parser.add_argument('--tf_log_level', type=int, default=0, help='Tensorflow C++ log level.')
	parser.add_argument('--slack_url', default=None, help='slack webhook notification destination link')
	args = parser.parse_args()

	accepted_models = ['Proposed', 'DNN', 'bDNN', 'LSTM', 'ACAM']

	if args.model not in accepted_models:
		raise ValueError('please enter a valid model to train: {}'.format(accepted_models))

	log_dir, hparams = prepare_run(args)

	train(args, log_dir, hparams)


if __name__ == '__main__':
	main()
