import os
import time
import traceback
from datetime import datetime

import infolog
import numpy as np
import tensorflow as tf
from sklearn import metrics
from hparams import hparams_debug_string
from modules.feeder import Feeder
from modules.models import create_model
from modules.utils import ValueWindow
from tqdm import tqdm

log = infolog.log


def add_train_stats(model, hparams):
	with tf.variable_scope('stats') as scope:
		tf.summary.scalar('loss-postnet', model.postnet_loss)
		tf.summary.scalar('loss_pipenet', model.pipenet_loss)
		tf.summary.scalar('loss_attention', model.attention_loss)
		tf.summary.scalar('loss_total', model.total_loss)
		tf.summary.scalar('accuracy-postnet', model.postnet_accuracy)
		tf.summary.scalar('accuracy_pipenet', model.pipenet_accuracy)
		tf.summary.scalar('learning_rate', model.learning_rate)  # Control learning rate decay speed
		gradient_norms = [tf.norm(grad) for grad in model.gradients]
		tf.summary.histogram('gradient_norm', gradient_norms)
		tf.summary.histogram('postnet_prediction', model.postnet_prediction)
		tf.summary.histogram('targets', model.targets)
		tf.summary.histogram('pipenet_prediction', model.pipenet_prediction)
		tf.summary.histogram('pipenet_targets', model.pipenet_targets)
		tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))  # visualize gradients (in case of explosion)
		return tf.summary.merge_all()


def add_eval_stats(summary_writer, step, postnet_loss, pipe_loss, att_loss, total_loss, acc, pipe_acc, auc):
	values = [
		tf.Summary.Value(tag='VAD_eval_model/eval_stats/eval_post_loss', simple_value=postnet_loss),
		tf.Summary.Value(tag='VAD_eval_model/eval_stats/eval_pipe_loss', simple_value=pipe_loss),
		tf.Summary.Value(tag='VAD_eval_model/eval_stats/eval_attention_loss', simple_value=att_loss),
		tf.Summary.Value(tag='VAD_eval_model/eval_stats/eval_total_loss', simple_value=total_loss),
		tf.Summary.Value(tag='VAD_eval_model/eval_stats/eval_post_acc', simple_value=acc),
		tf.Summary.Value(tag='VAD_eval_model/eval_stats/eval_pipe_acc', simple_value=pipe_acc),
		tf.Summary.Value(tag='VAD_eval_model/eval_stats/eval_auc', simple_value=auc)
		]
	test_summary = tf.Summary(value=values)
	summary_writer.add_summary(test_summary, step)


def time_string():
	return datetime.now().strftime('%Y-%m-%d %H:%M')


def model_train_mode(args, feeder, hparams, global_step):
	with tf.variable_scope('VAD_model', reuse=tf.AUTO_REUSE) as scope:
		model_name = args.model
		model = create_model(model_name or args.model, hparams)
		model.initialize(feeder.inputs, feeder.targets, global_step=global_step, is_training=True)
		model.add_loss()
		model.add_optimizer(global_step)
		stats = add_train_stats(model, hparams)
		return model, stats


def model_test_mode(args, feeder, hparams, global_step):
	with tf.variable_scope('VAD_model', reuse=tf.AUTO_REUSE) as scope:
		model_name = args.model
		model = create_model(model_name or args.model, hparams)
		model.initialize(feeder.eval_inputs, feeder.eval_targets, global_step=global_step, is_training=False, is_evaluating=True)
		model.add_loss()
		return model


def train(log_dir, args, hparams):
	save_dir = os.path.join(log_dir, 'vad_pretrained')
	plot_dir = os.path.join(log_dir, 'plots')
	eval_dir = os.path.join(log_dir, 'eval-dir')
	eval_plot_dir = os.path.join(eval_dir, 'plots')
	tensorboard_dir = os.path.join(log_dir, 'vad_events')
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(plot_dir, exist_ok=True)
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(eval_plot_dir, exist_ok=True)
	os.makedirs(tensorboard_dir, exist_ok=True)

	checkpoint_path = os.path.join(save_dir, 'vad_model.ckpt')
	input_path = os.path.join(args.base_dir, args.vad_input)

	log('Checkpoint path: {}'.format(checkpoint_path))
	log('Loading training data from: {}'.format(input_path))
	log('Using model: {}'.format(args.model))
	log(hparams_debug_string())

	# Start by setting a seed for repeatability
	tf.set_random_seed(hparams.vad_random_seed)

	# Set up data feeder
	coord = tf.train.Coordinator()
	with tf.variable_scope('datafeeder') as scope:
		feeder = Feeder(coord, input_path, hparams)

	# Set up model:
	global_step = tf.Variable(0, name='global_step', trainable=False)
	model, stats = model_train_mode(args, feeder, hparams, global_step)
	eval_model = model_test_mode(args, feeder, hparams, global_step)

	# Book keeping
	step = 0
	time_window = ValueWindow(100)
	total_loss_window = ValueWindow(100)
	post_loss_window = ValueWindow(100)
	post_acc_window = ValueWindow(100)
	pipe_loss_window = ValueWindow(100)
	pipe_acc_window = ValueWindow(100)
	att_loss_window = ValueWindow(100)
	eval_total_loss_window = ValueWindow(100)
	eval_post_loss_window = ValueWindow(100)
	eval_pipe_loss_window = ValueWindow(100)
	eval_att_loss_window = ValueWindow(100)
	eval_post_acc_window = ValueWindow(100)
	eval_pipe_acc_window = ValueWindow(100)
	eval_auc_window = ValueWindow(100)
	prev_eval_loss = 2.
	saver = tf.train.Saver(max_to_keep=10)

	log('VAD training set to a maximum of {} steps'.format(args.vad_train_steps))

	# Memory allocation on the GPU as needed
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# Train
	with tf.Session(config=config) as sess:
		try:
			summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
			sess.run(tf.global_variables_initializer())

			# saved model restoring
			if args.restore:
				# Restore saved model if the user requested it, default = True
				try:
					checkpoint_state = tf.train.get_checkpoint_state(save_dir)

					if checkpoint_state and checkpoint_state.model_checkpoint_path:
						log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
						saver.restore(sess, checkpoint_state.model_checkpoint_path)
					else:
						log('No model to load at {}'.format(save_dir), slack=True)

				except tf.errors.OutOfRangeError as e:
					log('Cannot restore checkpoint: {}'.format(e), slack=True)
			else:
				log('Starting new training!', slack=True)

			# initializing feeder
			feeder.start_threads(sess)

			# Training loop
			while not coord.should_stop() and step < args.vad_train_steps:
				start_time = time.time()
				step, total_loss, post_loss, pipe_loss, opt, post_acc, pipe_acc, att_loss = sess.run([global_step,
																									  model.total_loss,
																									  model.postnet_loss,
																									  model.pipenet_loss,
																									  model.optimize,
																									  model.postnet_accuracy,
																									  model.pipenet_accuracy,
																									  model.attention_loss])
				time_window.append(time.time() - start_time)
				total_loss_window.append(total_loss)
				post_loss_window.append(post_loss)
				post_acc_window.append(post_acc)
				pipe_loss_window.append(pipe_loss)
				pipe_acc_window.append(pipe_acc)
				att_loss_window.append(att_loss)
				message = '[{:.2f} epoch] Step {:7d} [{:.3f} sec/step, avg_total_loss={:.5f}, avg_att_loss={:.5f}, ' \
						  'avg_post_loss={:.5f}, avg_pipe_loss={:.5f}, avg_post_acc={:.5f}, avg_pipe_acc={:.5f}]'.format(
					(step / feeder.train_steps), step, time_window.average, total_loss_window.average, att_loss_window.average,
					post_loss_window.average, pipe_loss_window.average, post_acc_window.average, pipe_acc_window.average)
				log(message, end='\r', slack=(step % args.checkpoint_interval == 0))

				if total_loss > 100 or np.isnan(total_loss):
					log('Loss exploded to {:.5f} at step {}'.format(total_loss, step))
					raise Exception('Loss exploded')

				if step % args.checkpoint_interval == 0:
					# Save model and current global step
					saver.save(sess, checkpoint_path, global_step=global_step)

				if step % args.summary_interval == 0:
					log('\nWriting summary at step {}'.format(step))
					summary_writer.add_summary(sess.run(stats), step)

				if step % args.eval_interval == 0:
					#Run eval and save eval stats
					log('\nRunning evaluation at step {}'.format(step))

					eval_total_losses = []
					eval_post_losses = []
					eval_pipe_losses = []
					eval_att_losses = []
					eval_post_accs = []
					eval_pipe_accs = []
					eval_aucs = []

					for i in tqdm(range(feeder.test_steps)):
						ep1loss, ep2loss, ep1acc, ep2acc, eattloss, etloss = sess.run([eval_model.postnet_loss, eval_model.pipenet_loss,
																	   eval_model.postnet_accuracy, eval_model.pipenet_accuracy,
																	   eval_model.attention_loss, eval_model.total_loss])
						# auc calculate
						eval_raw_labels, eval_soft_result = sess.run([eval_model.raw_labels, eval_model.soft_prediction])
						fpr, tpr, thresholds = metrics.roc_curve(eval_raw_labels, eval_soft_result, pos_label=1)
						eval_auc = metrics.auc(fpr, tpr)
						eval_post_losses.append(ep1loss)
						eval_pipe_losses.append(ep2loss)
						eval_att_losses.append(eattloss)
						eval_post_accs.append(ep1acc)
						eval_pipe_accs.append(ep2acc)
						eval_aucs.append(eval_auc)
						eval_total_losses.append(etloss)

					eval_post_loss = sum(eval_post_losses) / len(eval_post_losses)
					eval_pipe_loss = sum(eval_pipe_losses) / len(eval_pipe_losses)
					eval_att_loss = sum(eval_att_losses) / len(eval_att_losses)
					eval_total_loss = sum(eval_total_losses) / len(eval_total_losses)
					eval_post_acc = sum(eval_post_accs) / len(eval_post_accs)
					eval_pipe_acc = sum(eval_pipe_accs) / len(eval_pipe_accs)
					eval_auc = sum(eval_aucs) / len(eval_aucs)

					eval_post_loss_window.append(eval_post_loss)
					eval_pipe_loss_window.append(eval_pipe_loss)
					eval_att_loss_window.append(eval_att_loss)
					eval_post_acc_window.append(eval_post_acc)
					eval_pipe_acc_window.append(eval_pipe_acc)
					eval_auc_window.append(eval_auc)
					eval_total_loss_window.append(eval_total_loss)


					log('Saving eval log to {}..'.format(eval_dir))
					# Save some log to monitor model improvement on same unseen sequence
					log('Eval post loss for global step {}: {:.3f}'.format(step, eval_post_loss_window.average))
					log('Eval pipe loss for global step {}: {:.3f}'.format(step, eval_pipe_loss_window.average))
					log('Eval att loss for global step {}: {:.3f}'.format(step, eval_att_loss_window.average))
					log('Eval total loss for global step {}: {:.3f}'.format(step, eval_total_loss_window.average))
					log('Eval post acc for global step {}: {:.3f}'.format(step, eval_post_acc_window.average))
					log('Eval pipe acc for global step {}: {:.3f}'.format(step, eval_pipe_acc_window.average))
					log('Eval auc for global step {}: {:.3f}'.format(step, eval_auc_window.average))
					log('Writing eval summary!')
					add_eval_stats(summary_writer, step,
								   eval_post_loss_window.average,
								   eval_pipe_loss_window.average,
								   eval_att_loss_window.average,
								   eval_total_loss_window.average,
								   eval_post_acc_window.average,
								   eval_pipe_acc_window.average,
								   eval_auc_window.average)

					if eval_total_loss_window.average < prev_eval_loss:
						# Save model and current global step
						saver.save(sess, checkpoint_path, global_step=global_step)
						prev_eval_loss = eval_total_loss_window.average

				if step == args.vad_train_steps:
					# Save model and current global step
					saver.save(sess, checkpoint_path, global_step=global_step)

			log('VAD training complete after {} global steps!'.format(args.vad_train_steps), slack=True)
			return save_dir

		except Exception as e:
			log('Exiting due to exception: {}'.format(e), slack=True)
			traceback.print_exc()
			coord.request_stop(e)


def vad_train(args, log_dir, hparams):
	return train(log_dir, args, hparams)
