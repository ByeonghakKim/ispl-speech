import os
import threading
import time
import traceback

import random
import numpy as np
import tensorflow as tf
from infolog import log
from modules.utils import data_transform, data_transform_targets_bdnn

_num_per_batch = 8


class Feeder:
    """
        Feeds batches of data into queue on a background thread.
    """

    def __init__(self, coordinator, metadata_filename, hparams):
        super(Feeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        self._train_offset = 0
        self._test_offset = 0
        self._start_idx = self._hparams.w

        # Load metadata
        self._mel_dir = os.path.join(os.path.dirname(metadata_filename), 'mels')
        with open(metadata_filename, encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]
            timesteps = sum([int(x[2]) for x in self._metadata])
            sr = hparams.sample_rate
            hours = timesteps / sr / 3600
            log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(self._metadata), hours))

        # Train test split
        train_indices, test_indices = train_validation_split(self._metadata, test_size=hparams.vad_test_size, seed=hparams.vad_data_random_state)

        self._train_meta = list(np.array(self._metadata)[train_indices])
        self._test_meta = list(np.array(self._metadata)[test_indices])

        train_buffer = len(self._train_meta) % _num_per_batch

        self.train_steps = sum([(int(m[3]) - 2 * self._hparams.w) for m in self._train_meta]) // hparams.vad_batch_size + train_buffer
        self.test_steps = sum([(int(m[3]) - 2 * self._hparams.w) for m in self._test_meta]) // (hparams.vad_batch_size * _num_per_batch)

        if hparams.vad_test_size is None:
            assert hparams.vad_test_batches == self.test_steps

        with tf.device('/cpu:0'):
            # Create placeholders for inputs and targets. Don't specify batch size because we want
            # to be able to feed different batch sizes at eval time.
            self._placeholders = [
                tf.placeholder(tf.float32, shape=(None, int(2*(hparams.w-1)/hparams.u+3), hparams.num_mels), name='inputs'),
                tf.placeholder(tf.float32, shape=(None, int(2*(hparams.w-1)/hparams.u+3)), name='targets')
                ]

            dtypes = [tf.float32, tf.float32]

            # Create queue for buffering data
            queue = tf.FIFOQueue(8, dtypes, name='input_queue')
            self._enqueue_op = queue.enqueue(self._placeholders)
            self.inputs, self.targets = queue.dequeue()
            self.inputs.set_shape(self._placeholders[0].shape)
            self.targets.set_shape(self._placeholders[1].shape)

            # Create eval queue for buffering eval data
            eval_queue = tf.FIFOQueue(1, dtypes, name='eval_queue')
            self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
            self.eval_inputs, self.eval_targets = eval_queue.dequeue()
            self.eval_inputs.set_shape(self._placeholders[0].shape)
            self.eval_targets.set_shape(self._placeholders[1].shape)

    def start_threads(self, session):
        self._session = session
        thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
        thread.daemon = True # Thread will close when parent quits
        thread.start()

        thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
        thread.daemon = True # Thread will close when parent quits
        thread.start()

    def _get_test_groups(self):

        if self._test_offset >= len(self._test_meta):
            self._test_offset = 0

        meta = self._test_meta[self._test_offset]
        self._test_offset += 1

        start_frame = int(meta[4])
        end_frame = int(meta[5])

        mel_input = np.load(os.path.join(self._mel_dir, meta[1]))
        mel_input = np.divide((mel_input - mel_input.min(axis=0)), (mel_input.max(axis=0) - mel_input.min(axis=0)))
        target = np.asarray([0] * (len(mel_input)))
        target[start_frame:end_frame] = 1

        mel_input = data_transform(mel_input, self._hparams.w, self._hparams.u, mel_input.min())
        target = data_transform_targets_bdnn(target, self._hparams.w, self._hparams.u)
        mel_input = mel_input[self._hparams.w:-self._hparams.w, :, :]
        target = target[self._hparams.w:-self._hparams.w]

        return (mel_input, target)

    def make_test_batches(self):
        start = time.time()

        # Read a group of examples
        n = self._hparams.vad_batch_size * _num_per_batch

        # Test on entire test set
        examples = [self._get_test_groups() for i in range(len(self._test_meta))]

        # Bucket examples based on similar output sequence length for efficiency
        examples.sort(key=lambda x: len(x[-1]))
        examples = (np.vstack([ex[0] for ex in examples]), np.vstack([ex[1] for ex in examples]))
        batches = [(examples[0][i: i + n], examples[1][i: i + n]) for i in range(0, len(examples[-1]) + 1 - n, n)]
        if len(examples[-1]) % n != 0:
            batches.append((examples[0][-(len(examples[-1]) % n):], examples[1][-(len(examples[-1]) % n):]))
        self.test_steps = len(batches)
        log('\nGenerated {} test batches of size {} in {:.3f} sec'.format(sum([len(batch) for batch in batches]), n, time.time() - start))
        return batches

    def _enqueue_next_train_group(self):
        while not self._coord.should_stop():
            start = time.time()

            # Read a group of examples
            n = self._hparams.vad_batch_size
            examples = [self._get_next_example() for _ in range(_num_per_batch)]

            # Bucket examples based on similar output sequence length for efficiency
            examples.sort(key=lambda x: len(x[-1]))
            examples = (np.vstack([ex[0] for ex in examples]), np.vstack([ex[1] for ex in examples]))
            batches = [(examples[0][i: i+n], examples[1][i: i+n]) for i in range(0, len(examples[-1]) + 1 - n, n)]
            if len(examples[-1]) % n != 0:
                batches.append((examples[0][-(len(examples[-1]) % n):], examples[1][-(len(examples[-1]) % n):]))

            log('\nGenerated {} train batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
            for batch in batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
                self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def _enqueue_next_test_group(self):
        # Create test batches once and evaluate on them for all test steps
        test_batches = self.make_test_batches()
        while not self._coord.should_stop():
            for batch in test_batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
                self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self):
        """Gets a single example (input, mel_target, token_target, linear_target, mel_length) from_ disk
        """
        if self._train_offset >= len(self._train_meta):
            self._train_offset = 0
            np.random.shuffle(self._train_meta)

        meta = self._train_meta[self._train_offset]
        self._train_offset += 1

        start_frame = int(meta[4])
        end_frame = int(meta[5])

        mel_input = np.load(os.path.join(self._mel_dir, meta[1]))
        mel_input = np.divide((mel_input - mel_input.min(axis=0)), (mel_input.max(axis=0) - mel_input.min(axis=0)))
        target = np.asarray([0] * (len(mel_input)))
        target[start_frame:end_frame] = 1

        mel_input = data_transform(mel_input, self._hparams.w, self._hparams.u, mel_input.min())
        target = data_transform_targets_bdnn(target, self._hparams.w, self._hparams.u)
        mel_input = mel_input[self._hparams.w:-self._hparams.w, :, :]
        target = target[self._hparams.w:-self._hparams.w]

        return (mel_input, target)

    def _prepare_batch(self, batch):
        c = list(zip(batch[0], batch[1]))
        np.random.shuffle(c)
        inputs, targets = zip(*c)
        return (inputs, targets)


def train_validation_split(data, test_size=0.05, seed=0):
    random.seed(seed)
    training_idx = []
    validation_idx = []

    aug_list = [[idx, x] for idx, x in enumerate(data) if len(x[0].split('_')) > 4]
    snr_list = list(set([x[0].split('_')[5] for idx, x in aug_list]))
    noise_list = list(set([x[0].split('_')[4] for idx, x in aug_list]))
    clean_idx = list(range(len(data)))
    for idx,x in aug_list:
        clean_idx.remove(idx)

    random.shuffle(clean_idx, random.random)
    validation_split_idx = int(np.ceil(test_size * len(clean_idx)))
    training_idx += clean_idx[validation_split_idx:]
    validation_idx += clean_idx[0:validation_split_idx]

    meta_idx = {}
    for n in noise_list:
        meta_idx[n] = {}
        for s in snr_list:
            meta_idx[n][s] = [idx for idx, x in aug_list if x[0].split('_')[4] == n and x[0].split('_')[5] == s]
            random.shuffle(meta_idx[n][s])
            validation_split_idx = int(np.ceil(test_size * len(meta_idx[n][s])))
            training_idx += meta_idx[n][s][validation_split_idx:]
            validation_idx += meta_idx[n][s][0:validation_split_idx]

    if bool(set(training_idx) & set(validation_idx)):
        raise ValueError('Training and validation data are overlapped!')

    random.shuffle(training_idx)
    random.shuffle(validation_idx)

    return training_idx, validation_idx



