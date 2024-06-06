import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,dropout=0.1)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))
        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=2):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,dropout=0.1)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))
        return output, self.hidden


class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size, hidden_size):
        super(lstm_seq2seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size)

    def forward(self, input_batch,target_batch, training_prediction='recursive', teacher_forcing_ratio=0.5):
        batch_size = input_batch.shape[1]
        target_len = target_batch.shape[0]
        outputs = torch.zeros(target_len, batch_size, self.input_size)
        
        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_output, encoder_hidden = self.encoder(input_batch)
        # import pdb; pdb.set_trace()
        decoder_input = input_batch[-1, :, :]
        decoder_hidden = encoder_hidden
        if training_prediction == 'recursive':
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output

        elif training_prediction == 'teacher_forcing':
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                if random.random() < teacher_forcing_ratio:
                    decoder_input = target_batch[t, :, :]
                else:
                    decoder_input = decoder_output

        elif training_prediction == 'mixed_teacher_forcing':
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                if random.random() < teacher_forcing_ratio:
                    decoder_input = target_batch[t, :, :]
                else:
                    decoder_input = decoder_output

        return outputs
