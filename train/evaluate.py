import torch
import torch.nn as nn
from torch import optim
import pickle
import sys

if len(sys.argv) < 2 or  len(sys.argv) > 2:  
    print ("""\
This script will generate lyrics based on the provided seed word

Usage:  python evaluate.py <seed word>
""")
    sys.exit(0)

generate_args={
    "temperature": 1, #temperature - higher will increase diversity
    "words":200, #number of words to generate
    "log_interval":30,
    "save":"../models/model.pt",
    "seed_word":sys.argv[1]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, dataframe):
        self.dictionary = Dictionary() 
        lyrics = dataframe['text'].apply(self.pre_process)
        train, test, y_train, y_test = train_test_split(lyrics, dataframe['artist'], test_size=0.15, random_state=13)
        train, val, y_train, y_val = train_test_split(train, y_train, test_size=0.15, random_state=13)
        self.train = self.tokenize(train.str.cat(sep=' end_song '))
        self.valid = self.tokenize(val.str.cat(sep=' end_song '))
        self.test = self.tokenize(test.str.cat(sep=' end_song '))
        self.train_raw = train.str.cat(sep=' end_song ')
        self.valid_raw = val.str.cat(sep=' end_song ')
        self.test_raw = test.str.cat(sep=' end_song ')
        
    def pre_process(self,text):
        text = text.replace("\r"," ")
        text = text.replace("\n"," ")
        text = text.lower()
        table = str.maketrans('', '', '!"#$%&\()*+-/:;<=>?@[\\]^_`{|}~')
        text = text.translate(table)
        return(text)
    
    def tokenize(self, string):
        tokens = 0
        words = nltk.word_tokenize(string)
        tokens += len(words)
        for word in words:
            self.dictionary.add_word(word)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        words = nltk.word_tokenize(string)
        for word in words:
            ids[token] = self.dictionary.word2idx[word]
            token += 1

        return ids

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

f = open('../models/corpus.pkl','rb')
corpus = pickle.load(f)
f.close

with open(generate_args['save'], 'rb') as f:
    model = torch.load(f,map_location='cpu').to(device)
model.eval()
seed_word = generate_args['seed_word']
seed=torch.LongTensor(1,1).to(device)
seed[0]=corpus.dictionary.word2idx[seed_word]
hidden = model.init_hidden(1)
input = seed
gen_text=""

gen_text = gen_text + seed_word + ' '
with torch.no_grad():  # no tracking history
	for i in range(generate_args['words']):
	    output, hidden = model(input, hidden)
	    word_weights = output.squeeze().div(generate_args['temperature']).exp().cpu()
	    word_idx = torch.multinomial(word_weights, 1)[0]
	    input.fill_(word_idx)
	    word = corpus.dictionary.idx2word[word_idx]
	    
	    gen_text = gen_text + word + ' '

	    if i % generate_args['log_interval'] == 0:
	       print('| Generated {}/{} words'.format(i, generate_args['words']))

print("="*89)
print("generated text ::\n"+gen_text)
