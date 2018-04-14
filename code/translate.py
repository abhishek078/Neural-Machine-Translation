import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_utils import process_data
from data_utils import split_data
from data_utils import generate_epoch
from data_utils import generate_batch
from model import model

class hyper_params(object):
	def __init__(self):
		self.num_epochs = 512
		self.batch_size = 4
		self.num_hidden_units = 128
		self.num_layers = 1
		self.dropout = 0.0

def train(params):
	hindi_token_ids, hindi_seq_lens, hindi_vocab_dict, hindi_rev_vocab_dict = process_data('../data/hindi_dump.p', max_vocab_size=100000, target_lang=False)
	bengali_token_ids, bengali_seq_lens, bengali_vocab_dict, bengali_rev_vocab_dict = process_data('../data/bengali_dump.p', max_vocab_size=100000, target_lang=True)
	train_encoder_inputs, train_decoder_inputs, train_targets, train_hindi_seq_lens, train_bengali_seq_len, valid_encoder_inputs, valid_decoder_inputs, valid_targets, valid_hindi_seq_lens, valid_bengali_seq_lens = split_data(hindi_token_ids, bengali_token_ids, hindi_seq_lens, bengali_seq_lens,train_ratio=0.8)

	params.hindi_vocab_size = len(hindi_vocab_dict)
	params.bengali_vocab_size = len(bengali_vocab_dict)

	print params.hindi_vocab_size, params.bengali_vocab_size

	with tf.Session() as sess:
		_model = model(params)
		sess.run(tf.global_variables_initializer())
		losses = []
		accs = []
		for epoch_num, epoch in enumerate(generate_epoch(train_encoder_inputs,train_decoder_inputs, train_targets,train_hindi_seq_lens, train_bengali_seq_len,params.num_epochs, params.batch_size)):
			print "EPOCH : ", epoch_num
			sess.run(tf.assign(_model.lr, 0.01 * (0.99 ** epoch_num)))
			batch_loss = []
			batch_acc = []
			for batch_num, (batch_encoder_inputs, batch_decoder_inputs,batch_targets, batch_hindi_seq_lens,batch_bengali_seq_lens) in enumerate(epoch):
				loss, _,acc = _model.step(sess, params,batch_encoder_inputs, batch_decoder_inputs, batch_targets,batch_hindi_seq_lens, batch_bengali_seq_lens,params.dropout)
				batch_loss.append(loss)
				batch_acc.append(acc)
			losses.append(np.mean(batch_loss))
			accs.append(np.mean(batch_acc))
			print "Training Loss: ",losses[-1]
			print "Training Accuracy",accs[-1]
		plt.plot(losses, label='loss')
		plt.legend()
		# plt.show()
		
		plt.title('Plot for Training Error versus Epochs', fontsize='20', style='oblique')
		plt.xlabel('Epochs', fontsize='16', color='green')
		plt.ylabel('Training Error', fontsize='16', color='green')
		plt.savefig('../output/plot.png')
		plt.show()

		acc = _model.test(sess, params, valid_encoder_inputs, valid_decoder_inputs, valid_targets, valid_hindi_seq_lens, valid_bengali_seq_lens, params.dropout)
		print acc
		# print "logits"
		# print log,log.shape,len(log),len(log[0]),"\n\n"
		# print "target"
		# print tar,tar.shape,len(tar),len(tar[0]),"\n\n"
		# print "target_flat"
		# print tarf,tarf.shape,len(tarf),"\n\n"
		# print "output"
		# print out,len(out)

params = hyper_params()
train(params)
