import numpy as np
import tensorflow as tf
from infolog import log
from modules.utils import data_transform_targets_bdnn


def bdnn_prediction(batch_size_in, logits, threshold=0.5, w=19, u=9):
    bdnn_batch_size = batch_size_in + 2 * w
    result = np.zeros((int(bdnn_batch_size), 1))
    indx = np.arange(int(bdnn_batch_size)) + 1
    indx = data_transform_targets_bdnn(indx, w, u)
    indx = indx[w:(int(bdnn_batch_size) - w), :]
    indx_list = np.arange(w, int(bdnn_batch_size) - w)

    for i in indx_list:
        indx_temp = np.where((indx - 1) == i)
        pred = logits[indx_temp]
        pred = np.sum(pred) / pred.shape[0]
        result[i] = pred

    result = result[w:-w]
    soft_result = np.float32(result)
    result = np.float32(result) >= threshold

    return result.astype(np.float32), soft_result


class spec_conv:
    """Two fully connected layers used as an information bottleneck for the attention.
    """
    def __init__(self, is_training, hparams, scope=None):
        """
        Args:
            is_training: Boolean. Controller of mechanism for batch normalization.
            hparams: Hyper parameters
            scope: Spectral attention scope.
        """
        super(spec_conv, self).__init__()
        self.filter_width = hparams.filter_width
        self.layers = hparams.layers
        self.conv_channels = hparams.conv_channels
        self.is_training = is_training

        self.scope = 'spectral_convolution' if scope is None else scope

        self.conv_layers = []
        with tf.variable_scope('spectral_attention_blocks'):
            for layer in range(self.layers):
                self.conv_layers.append([tf.layers.Conv2D(filters=(self.conv_channels * (2**layer)),
                                                          kernel_size=self.filter_width,
                                                          padding='same',
                                                          name='block_linear_conv_{}'.format(layer+1)),
                                         tf.layers.Conv2D(filters=(self.conv_channels * (2**layer)),
                                                          kernel_size=self.filter_width,
                                                          padding='same',
                                                          name='block_sigmoid_conv_{}'.format(layer+1))])

    def __call__(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        num_time = x.get_shape().as_list()[1]
        conv_list = list()
        with tf.variable_scope(self.scope):
            for conv_linear, conv_sigmoid in self.conv_layers:
                x_linear = conv_linear(x)
                x_sigmoid = conv_sigmoid(x)
                x_linear = tf.layers.batch_normalization(x_linear, training=self.is_training)
                x_sigmoid = tf.layers.batch_normalization(x_sigmoid, training=self.is_training)
                x = tf.sigmoid(x_sigmoid) * x_linear
                x = tf.layers.max_pooling2d(x, pool_size=(1, 2), strides=(1, 2), padding='same')
                conv_list.append(x)
            x = tf.reshape(x, (-1, num_time, x.get_shape().as_list()[2] * x.get_shape().as_list()[3]))
        return x, conv_list


class Prenet:
    """Two fully connected layers used as an information bottleneck for the attention.
    """
    def __init__(self, is_training, layers_sizes=[256, 256], drop_rate=0.5, activation=tf.nn.relu, pipenet=False, scope=None):
        """
        Args:
            layers_sizes: list of integers, the length of the list represents the number of pre-net
                layers and the list values represent the layers number of units
            activation: callable, activation functions of the prenet layers.
            scope: Prenet scope.
        """
        super(Prenet, self).__init__()
        self.drop_rate = drop_rate

        self.layers_sizes = layers_sizes
        self.activation = activation
        self.is_training = is_training
        self.pipenet = pipenet

        self.scope = 'prenet' if scope is None else scope

    def __call__(self, inputs):
        x = inputs
        with tf.variable_scope(self.scope):
            for i, size in enumerate(self.layers_sizes):
                dense = tf.layers.dense(x, units=size, activation=None, name='dense_{}'.format(i + 1))
                dense = tf.layers.batch_normalization(dense, training=self.is_training)
                dense = self.activation(dense)
                x = tf.layers.dropout(dense, rate=self.drop_rate, training=self.is_training, name='dropout_{}'.format(i + 1) + self.scope)
            if self.pipenet:
                return x, tf.squeeze(tf.layers.dense(x, units=1, activation=None, name='pipenet'), axis=-1)
            else:
                return x


class Postnet:
    """Two fully connected layers used as an information bottleneck for the attention.
    """
    def __init__(self, is_training, layers_sizes=[256, 1], drop_rate=0.5, activation=tf.nn.relu, scope=None):
        """
        Args:
            layers_sizes: list of integers, the length of the list represents the number of pre-net
                layers and the list values represent the layers number of units
            activation: callable, activation functions of the postnet layers.
            scope: Postnet scope.
        """
        super(Postnet, self).__init__()
        self.drop_rate = drop_rate

        self.layers_sizes = layers_sizes
        self.activation = activation
        self.is_training = is_training

        self.scope = 'postnet' if scope is None else scope

    def __call__(self, inputs):
        x = inputs
        with tf.variable_scope(self.scope):
            for i, size in enumerate(self.layers_sizes):
                dense = tf.layers.dense(x, units=size, activation=None, name='dense_{}'.format(i + 1))
                if i < len(self.layers_sizes) - 1:
                    dense = tf.layers.batch_normalization(dense, training=self.is_training)
                    dense = self.activation(dense)
                    x = tf.layers.dropout(dense, rate=self.drop_rate, training=self.is_training, name='dropout_{}'.format(i + 1) + self.scope)
                else:
                    x = dense
        return tf.squeeze(x, axis=-1)


def smooth_softmax(x):
    return tf.sigmoid(x) / tf.expand_dims(tf.reduce_sum(tf.sigmoid(x), axis=-1), axis=-1)


# Soft Attention function
def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        queries = tf.reduce_mean(queries, axis=1)[:, tf.newaxis, :]
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Activation
        attention_weights = tf.nn.softmax(outputs, axis=-1)  # (h*N, T_q, T_k)

        # Apply attention
        outputs = tf.multiply(tf.squeeze(attention_weights, axis=1)[:, :, tf.newaxis], V_)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
        attention_weights = tf.concat(tf.split(attention_weights, num_heads, axis=0), axis=1)

    return outputs, attention_weights


def inference(inputs, is_training=True, hparams=None):

    with tf.variable_scope('proposed'):
        # Encoder
        with tf.variable_scope("masking"):
            # Soft attention (masking) at frequency axis
            spec_att = spec_conv(is_training=is_training, hparams=hparams)
            spec_att_output, conv_list = spec_att(inputs)
        # Pipenet
        with tf.variable_scope("pipenet"):
            pipenet = Prenet(is_training=is_training, layers_sizes=[hparams.prenet_units, hparams.prenet_units], pipenet=True)
            pipenet_output, midnet_output = pipenet(spec_att_output)
        # Multi-head Attention
        with tf.variable_scope("attention"):
            z, alpha = multihead_attention(pipenet_output, pipenet_output, hparams.num_att_units, hparams.num_heads)
        # Postnet
        with tf.variable_scope("postnet"):
            ## Postnet
            postnet = Postnet(is_training=is_training, layers_sizes=[hparams.prenet_units, 1])
            postnet_output = postnet(z)

        return midnet_output, postnet_output, alpha, conv_list


class Proposed():
    """Proposed Feature prediction Model.
    """
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, targets, global_step=None, is_training=False, is_evaluating=False):
        """
        Initializes the model for inference


        Args:
            - inputs: float32 Tensor with shape [N, hparams.num_mels, 1+2*hparams.num_slide]
            - targets: float32 Tensor with shape [N]
        """
        if inputs is None:
            raise ValueError('no mel inputs were provided')
        if targets is None:
            raise ValueError('no targets were provided')
        if is_training and is_evaluating:
            raise RuntimeError('Model can not be in training and evaluation modes at the same time!')

        with tf.variable_scope('inference') as scope:
            self.batch_size = tf.shape(inputs)[0]
            hp = self._hparams

            self.pipenet_output, self.postnet_output, self.alpha, self.conv_list = inference(inputs, is_training=is_training, hparams=hp)

            _, self.soft_prediction = self.bdnn_prediction(tf.sigmoid(self.postnet_output))

            self.pipenet_prediction = tf.round(tf.sigmoid(self.pipenet_output))

            self.postnet_prediction = tf.round(tf.sigmoid(self.postnet_output))

            self.all_vars = tf.trainable_variables()

            self.inputs = inputs
            self.pipenet_targets = targets
            self.targets = tf.reduce_max(targets, axis=-1)

            raw_indx = int(np.floor(int(2 * (self._hparams.w - 1) / self._hparams.u + 3) / 2))
            raw_labels = self.pipenet_targets[:, raw_indx]
            raw_labels = tf.reshape(raw_labels, shape=(-1, 1))
            self.raw_labels = tf.identity(raw_labels, 'raw_labels')

            self.postnet_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.postnet_prediction, self.pipenet_targets), tf.float32))
            self.pipenet_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pipenet_prediction, self.pipenet_targets), tf.float32))

            log('Initialized Proposed model. Dimensions (? = dynamic shape): ')
            log('  Train mode:               {}'.format(is_training))
            log('  Eval mode:                {}'.format(is_evaluating))
            log('  input:                    {}'.format(inputs.shape))
            log('  Parameters                {:.3f} Million.'.format(np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1_000_000))

    def bdnn_prediction(self, logits, threshold=0.5):
        th_tensor = tf.constant(threshold, dtype=tf.float32)
        result, soft_result = tf.py_func(bdnn_prediction, [self.batch_size, logits, th_tensor], Tout=[tf.float32, tf.float32])
        return result, soft_result

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hparams
            self.postnet_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.pipenet_targets, logits=self.postnet_output))
            self.pipenet_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.pipenet_targets, logits=self.pipenet_output))
            self.attention_loss = 0.1 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.pipenet_targets, logits=tf.reduce_max(self.alpha, axis=1)))

            # Regularize variables
            self.regularization = tf.add_n([tf.nn.l2_loss(v) for v in self.all_vars if not('bias' in v.name or 'Bias' in v.name)]) * hp.vad_reg_weight

            self.total_loss = self.postnet_loss + self.pipenet_loss + self.attention_loss + self.regularization

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
            global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hparams
            if hp.vad_decay_learning_rate:
                self.decay_steps = hp.vad_decay_steps
                self.decay_rate = hp.vad_decay_rate
                self.learning_rate = self._learning_rate_decay(hp.vad_initial_learning_rate, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.vad_initial_learning_rate)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.vad_adam_beta1, hp.vad_adam_beta2, hp.vad_adam_epsilon)
            gradients, variables = zip(*optimizer.compute_gradients(self.total_loss, self.all_vars))
            self.gradients = gradients
            #Just for causion
            #https://github.com/Rayhane-mamah/Tacotron-2/issues/11
            if hp.vad_clip_gradients:
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.)
            else:
                clipped_gradients = gradients
            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)

    def _learning_rate_decay(self, init_lr, global_step):
        #################################################################
        # Narrow Exponential Decay:

        # Phase 1: lr = 1e-3
        # We only start learning rate decay after 50k steps

        # Phase 2: lr in ]1e-5, 1e-3[
        # decay reach minimal value at step 310k

        # Phase 3: lr = 1e-5
        # clip by minimal learning rate value (step > 310k)
        #################################################################
        hp = self._hparams
        #Compute natural exponential decay
        lr = tf.train.exponential_decay(init_lr,
            global_step - hp.vad_start_decay, #lr = 1e-3 at step 50k
            self.decay_steps,
            self.decay_rate,
            name='lr_exponential_decay')
        #clip learning rate by max and min values (initial and final values)
        return tf.minimum(tf.maximum(lr, hp.vad_final_learning_rate), init_lr)
