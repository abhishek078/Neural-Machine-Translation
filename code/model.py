import tensorflow as tf

def fn_softmax(params, outputs, scope):
	with tf.variable_scope(scope, reuse=True):
		W_softmax = tf.get_variable("W_softmax", [params.num_hidden_units, params.bengali_vocab_size])
		b_softmax = tf.get_variable("b_softmax", [params.bengali_vocab_size])
	logits = tf.matmul(outputs, W_softmax) + b_softmax
	return logits

class model(object):
	def __init__(self, params):
		self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name='encoder_inputs')
		self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name='decoder_inputs')
		self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')
		self.hindi_seq_lens = tf.placeholder(tf.int32, shape=[None, ], name="hindi_seq_lens")
		self.bengali_seq_lens = tf.placeholder(tf.int32, shape=[None, ], name="bengali_seq_lens")
		self.dropout = tf.placeholder(tf.float32)

		with tf.variable_scope('encoder') as scope:
			W_input = tf.get_variable("W_input", [params.hindi_vocab_size, params.num_hidden_units])
			self.embedded_encoder_inputs = tf.nn.embedding_lookup(W_input, self.encoder_inputs)
			# Forward direction cell
			lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(params.num_hidden_units, forget_bias=1.0)
			# Backward direction cell
			lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(params.num_hidden_units, forget_bias=1.0)
			self.encoder_outputs, self.encoder_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_encoder_inputs, dtype=tf.float32)

		with tf.variable_scope('decoder') as scope:
			h1 = tf.concat([self.encoder_state[0][0], self.encoder_state[1][0]], 0)
			c1 = tf.concat([self.encoder_state[0][1], self.encoder_state[1][1]], 0)
			self.decoder_initial_state = (self.encoder_state[0],)
			single_cell = tf.nn.rnn_cell.BasicLSTMCell(params.num_hidden_units)
			single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell)
			self.decoder_stacked_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * params.num_layers)
			W_input = tf.get_variable("W_input", [params.bengali_vocab_size, params.num_hidden_units])
			self.embedded_decoder_inputs = embeddings = tf.nn.embedding_lookup(W_input, self.decoder_inputs)

			self.all_decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(cell=self.decoder_stacked_cell,inputs=self.embedded_decoder_inputs,sequence_length=self.bengali_seq_lens, time_major=False,	initial_state=self.decoder_initial_state)
			W_softmax = tf.get_variable("W_softmax", [params.num_hidden_units, params.bengali_vocab_size])
			b_softmax = tf.get_variable("b_softmax", [params.bengali_vocab_size])

			self.decoder_outputs_flat = tf.reshape(self.all_decoder_outputs, [-1, params.num_hidden_units])
			self.logits_flat = fn_softmax(params, self.decoder_outputs_flat, scope=scope)
			self.targets_flat = tf.reshape(self.targets, [-1])

			self.correct_prediction = tf.equal(tf.argmax(self.logits_flat, 1), tf.cast(self.targets_flat, tf.int64))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

			tf.summary.scalar('accuracy', self.accuracy)
			losses_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_flat, labels=self.targets_flat)
			mask = tf.sign(tf.to_float(self.targets_flat))
			masked_losses = (mask * losses_flat)
			masked_losses = tf.reshape(masked_losses, tf.shape(self.targets))
			self.loss = tf.reduce_mean(tf.reduce_sum(masked_losses, reduction_indices=1))
		self.lr = tf.Variable(0.0, trainable=False)
		trainable_vars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 5.0)
		optimizer = tf.train.RMSPropOptimizer(self.lr)
		self.train_optimizer = optimizer.apply_gradients(zip(grads, trainable_vars))

	def step(self, sess, params, batch_encoder_inputs, batch_decoder_inputs, batch_targets, batch_hindi_seq_lens, batch_bengali_seq_lens, dropout):
		input_feed = {self.encoder_inputs: batch_encoder_inputs, self.decoder_inputs: batch_decoder_inputs, self.targets: batch_targets, self.hindi_seq_lens: batch_hindi_seq_lens, self.bengali_seq_lens: batch_bengali_seq_lens, self.dropout: dropout}
		output_feed = [self.loss, self.train_optimizer]
		outputs = sess.run(output_feed, input_feed)
		acc = sess.run([self.accuracy], input_feed)
		return outputs[0], outputs[1],acc


	def test(self, sess, params, batch_encoder_inputs, batch_decoder_inputs, batch_targets, batch_hindi_seq_lens, batch_bengali_seq_lens, dropout):
		input_feed = {self.encoder_inputs: batch_encoder_inputs, self.decoder_inputs: batch_decoder_inputs, self.targets: batch_targets, self.hindi_seq_lens: batch_hindi_seq_lens, self.bengali_seq_lens: batch_bengali_seq_lens, self.dropout: dropout}
		output_feed = [self.accuracy]
		accuracy = sess.run(output_feed, input_feed)
		return accuracy
