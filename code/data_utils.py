import sys
import re
import cPickle
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(BR"\d")

def basic_tokenizer(sentence):
	words = []
	for space_separated_item in sentence.strip().split():
		words.extend(_WORD_SPLIT.split(space_separated_item))
	return [w for w in words if w]

def get_vocab(tokenized, max_vocab_size):
	vocab = {}
	for sentence in tokenized:
		for word in sentence:
			if word in vocab:
				vocab[word] += 1
			else:
				vocab[word] = 1
	vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
	if len(vocab_list) > max_vocab_size:
		vocab_list = vocab_list[:max_vocab_size]

	vocab_dict = dict([(x,y) for (y,x) in enumerate(vocab_list)])
	rev_vocab_dict = {v: k for k, v in vocab_dict.iteritems()}

	return vocab_list, vocab_dict, rev_vocab_dict

def sentence_to_token_ids(sentence, vocab_dict, target_lang, normalize_digits=True):
	if not normalize_digits:
		tokens = [vocab_dict.get(w, UNK_ID) for w in sentence]
	else:
		tokens = [vocab_dict.get(_DIGIT_RE.sub(b"0", w), UNK_ID)
			for w in sentence]

	if target_lang:
		tokens.append(EOS_ID)

	return tokens


def data_to_token_ids(tokenized, vocab_dict, target_lang, normalize_digits=True):
	data_as_tokens = []
	seq_lens = []
	max_len = max(len(sentence) for sentence in tokenized) + 1

	for sentence in tokenized:
		token_ids = sentence_to_token_ids(sentence, vocab_dict, target_lang, normalize_digits)
		data_as_tokens.append(token_ids + [PAD_ID]*(max_len - len(token_ids)))
		seq_lens.append(len(token_ids))
	return np.array(data_as_tokens), np.array(seq_lens)

def process_data(datafile, max_vocab_size, target_lang):
	with open(datafile, 'rb') as f:
		sentences = cPickle.load(f)

	tokenized = []
	for i in xrange(len(sentences)):
		tokenized.append(basic_tokenizer(sentences[i]))

	vocab_list, vocab_dict, rev_vocab_dict = get_vocab(tokenized, max_vocab_size)

	data_as_tokens, seq_lens = data_to_token_ids(tokenized, vocab_dict, target_lang, normalize_digits=True)

	return data_as_tokens, seq_lens, vocab_dict, rev_vocab_dict

def split_data(hindi_token_ids, bengali_token_ids, hindi_seq_lens, bengali_seq_len, train_ratio=0.8):
	decoder_inputs = []
	targets = []
	for sentence in bengali_token_ids:
		decoder_inputs.append(np.array([GO_ID] + list(sentence)))
		targets.append(np.array(([GO_ID] + list(sentence))[1:] + [0]))

	bengali_token_ids = np.array(decoder_inputs)
	targets = np.array(targets)

	last_train_index = int(0.8*len(hindi_token_ids))

	train_encoder_inputs = hindi_token_ids[:last_train_index]
	train_decoder_inputs = bengali_token_ids[:last_train_index]
	train_targets = targets[:last_train_index]
	train_hindi_seq_lens = hindi_seq_lens[:last_train_index]
	train_bengali_seq_len = bengali_seq_len[:last_train_index]

	test_encoder_inputs = hindi_token_ids[last_train_index:]
	test_decoder_inputs = bengali_token_ids[last_train_index:]
	test_targets = targets[last_train_index:]
	test_hindi_seq_lens = hindi_seq_lens[last_train_index:]
	test_bengali_seq_len = bengali_seq_len[last_train_index:]

	return train_encoder_inputs, train_decoder_inputs, train_targets, train_hindi_seq_lens, train_bengali_seq_len, test_encoder_inputs, test_decoder_inputs, test_targets, test_hindi_seq_lens, test_bengali_seq_len

def generate_epoch(encoder_inputs, decoder_inputs, targets, hindi_seq_lens, bengali_seq_lens, num_epochs, batch_size):
	for epoch_num in range(num_epochs):
		yield generate_batch(encoder_inputs, decoder_inputs, targets, hindi_seq_lens, bengali_seq_lens, batch_size)

def generate_batch(encoder_inputs, decoder_inputs, targets,	hindi_seq_lens, bengali_seq_lens, batch_size):
	data_size = len(encoder_inputs)
	num_batches = (data_size // batch_size)
	for batch_num in range(num_batches):
		start_index = batch_num * batch_size
		end_index = min((batch_num + 1) * batch_size, data_size)
		yield encoder_inputs[start_index:end_index], decoder_inputs[start_index:end_index], targets[start_index:end_index],	hindi_seq_lens[start_index:end_index],	bengali_seq_lens[start_index:end_index]