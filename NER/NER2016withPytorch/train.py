import itertools
import optparse
import os
from collections import OrderedDict

import loader
from NER.NER2016withPytorch.evaluate import bilstm_train_and_eval
from models.config import LSTMConfig

# Read parameters from command line
from NER.NER2016withPytorch.utils import save_mappings

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    "-m", "--mapping_path", default="",
    help="Save the mappings"
)
opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method

# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert os.path.isfile(opts.mapping_path)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation script / folders
# if not os.path.isfile(eval_script):
#     raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
# if not os.path.exists(eval_temp):
#     os.makedirs(eval_temp)
# if not os.path.exists(models_path):
#     os.makedirs(models_path)


##############################
# Data Preprocessing as tagger
##############################

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)

# Use selected tagging scheme (IOB / IOBES)
loader.update_tag_scheme(train_sentences, tag_scheme)
loader.update_tag_scheme(dev_sentences, tag_scheme)
loader.update_tag_scheme(test_sentences, tag_scheme)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_words_train = loader.word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = loader.augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = loader.word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# print("*****Train Sentences[0]*****\n", train_sentences[0])
# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = loader.char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = loader.tag_mapping(train_sentences)

# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
if LSTMConfig.for_crf:
    word_to_id, tag_to_id, char_to_id = loader.extend_maps(word_to_id, tag_to_id, char_to_id)
else:
    word_to_id, tag_to_id, char_to_id = loader.extend_maps(word_to_id, tag_to_id, char_to_id, for_crf=False)

# Index data
train_data = loader.prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = loader.prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = loader.prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

# print("*****train_data[0] *****", train_data[0])
# 还需要额外的一些数据处理
train_data = loader.prepocess_data_for_lstmcrf(train_data, word_to_id, tag_to_id, char_to_id)
dev_data = loader.prepocess_data_for_lstmcrf(dev_data, word_to_id, tag_to_id, char_to_id)
test_data = loader.prepocess_data_for_lstmcrf(test_data, word_to_id, tag_to_id, char_to_id, test=True)
print("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

# Save the mappings to disk
print('Saving the mappings to disk...')
save_mappings(opts.mapping_path, id_to_word, id_to_char, id_to_tag)

# print(tag_to_id)
# {'O': 0, 'S-LOC': 1, 'B-PER': 2, 'E-PER': 3, 'S-ORG': 4, 'S-MISC': 5, 'B-ORG': 6, 'E-ORG': 7, 'S-PER': 8,
# 'I-ORG': 9, 'B-LOC': 10, 'E-LOC': 11, 'B-MISC': 12, 'E-MISC': 13, 'I-MISC': 14, 'I-PER': 15, 'I-LOC': 16,
# '<unk>': 17, '<pad>': 18, '<start>': 19, '<end>': 20}

# print(train_data[0])
# Dataset's format
# {
# 'str_words': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
# 'words': [963, 22406, 235, 770, 6, 4585, 209, 7683, 1],
# 'chars': [[35, 57], [6, 0, 61, 0, 11, 2, 7], [51, 0, 6, 13, 1, 3], [11, 1, 8, 8], [2, 5], [20, 5, 18, 11, 5, 2, 2],
#          [43, 6, 4, 2, 4, 7, 10], [8, 1, 13, 20], [17]],
# 'caps': [1, 0, 2, 0, 0, 0, 2, 0, 0],
# 'tags': [4, 0, 5, 0, 0, 0, 5, 0, 0]
# }

############################################################################
# Using pytorch instead of theano to realize the same architecture of tagger
############################################################################

lstmcrf_pred = bilstm_train_and_eval(
    train_data,
    dev_data,
    test_data,
    word_to_id, tag_to_id,
    id_to_word, id_to_tag
)


