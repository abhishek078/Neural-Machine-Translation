import sys
import cPickle

reload(sys)
sys.setdefaultencoding('utf8')

def create_data_dump():
	hindi_data_file_name = '../data/hindi_sentences.txt'
	bengali_data_file_name = '../data/bengali_sentences.txt'

	with open(hindi_data_file_name, 'rb') as f:
		hindi_text = f.read()
		hindi_text = hindi_text.decode("utf-8")

	hindi_sentences = hindi_text.split("\n")#nltk.tokenize.sent_tokenize(text)

	with open(bengali_data_file_name, 'rb') as f:
		bengali_text = f.read()
		bengali_text = bengali_text.decode("utf-8")

	bengali_sentences = bengali_text.split("\n")#nltk.tokenize.sent_tokenize(text)
	print "Hindi sentences : ", str(len(hindi_sentences)), ", Bengali sentences ", str(len(bengali_sentences))

	with open('../data/hindi_dump.p', 'wb') as f:
		cPickle.dump(hindi_sentences, f)
	with open('../data/bengali_dump.p', 'wb') as f:
		cPickle.dump(bengali_sentences, f)

create_data_dump()
