import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class Attention(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, attention_size, return_attention=False):
        """ Initialize the attention layer

        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction

        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.attention_vector = Parameter(torch.FloatTensor(attention_size))
        self.attention_vector.data.normal_(std=0.05) # Initialize attention vector

    def __repr__(self):
        s = '{name}({attention_size}, return attention={return_attention})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths):
        """ Forward pass.

        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences

        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len).cuda()).unsqueeze(0)
        mask = torch.tensor((idxes < input_lengths.unsqueeze(1)).float())

        # apply mask and renormalize attention scores (weights)
        masked_weights = unnorm_ai * mask
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)

        return (representations, attentions if self.return_attention else None)

class Torchmoji(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nb_classes, nb_tokens, feature_output=False,
                 embed_dropout_rate=0, final_dropout_rate=0, return_attention=False):
        super(Torchmoji, self).__init__()

        embedding_dim = 128
        hidden_size = 256
        nlayers = 2
        attention_size = 4 * hidden_size + embedding_dim

        self.feature_output = feature_output
        self.embed_dropout_rate = embed_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.return_attention = return_attention
        self.hidden_size = hidden_size
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.nlayers = nlayers

        self.add_module('embed', nn.Embedding(nb_tokens, embedding_dim))
        self.add_module('embed_dropout', nn.Dropout2d(embed_dropout_rate))
        self.add_module('lstm_0', nn.LSTM(embedding_dim, hidden_size, nlayers, bidirectional=True))
        self.add_module('lstm_1', nn.LSTM(hidden_size * 2, hidden_size, nlayers, bidirectional=True))
        self.add_module('attention_layer', Attention(attention_size=attention_size, return_attention=return_attention))
        if not feature_output:
            self.add_module('final_dropout', nn.Dropout(final_dropout_rate))
            self.add_module('output_layer', nn.Sequential(nn.Linear(attention_size, nb_classes if self.nb_classes > 2 else 1), nn.ReLU()))


    def forward(self, input_seqs, data_len):
        #pack inputs
        packed_input = pack_padded_sequence(input_seqs, data_len, batch_first=True)

        x = self.embed(packed_input.data)
        x = nn.Tanh()(x)
        x = self.embed_dropout(x)
        packed_input = PackedSequence(x, packed_input.batch_sizes)

        lstm_0_output, _ = self.lstm_0(packed_input)
        lstm_1_output, _ = self.lstm_1(lstm_0_output)
        packed_input = PackedSequence(torch.cat((lstm_1_output.data,
                                                 lstm_0_output.data,
                                                 packed_input.data), dim=1),
                                      packed_input.batch_sizes)
        input_seqs, _ = pad_packed_sequence(packed_input, batch_first=True)

        x, att_weights = self.attention_layer(input_seqs, data_len)
        if not self.feature_output:
            x = self.final_dropout(x)
            outputs = self.output_layer(x)
        else:
            outputs = outputs

        return outputs

