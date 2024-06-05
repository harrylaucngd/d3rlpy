import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        
        # Encoder layers
        self.enc_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.enc_bn1 = nn.BatchNorm1d(input_dim)
        self.enc_nonlinear = nn.ReLU()  # Non-linear activation
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.enc_bn2 = nn.BatchNorm1d(latent_dim)
        self.dropout = nn.Dropout(p=0.3)
        
        # Decoder layers
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.dec_bn1 = nn.BatchNorm1d(hidden_dim)
        self.dec_nonlinear = nn.ReLU()  # Non-linear activation
        self.dec_lstm = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.dec_bn2 = nn.BatchNorm1d(input_dim)

    def encode(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.reshape(batch_size * seq_len, -1)
        x = self.enc_bn1(x)
        x = x.reshape(batch_size, seq_len, -1)
        _, (h, _) = self.enc_lstm(x)
        h = h.squeeze(0)
        h = self.enc_nonlinear(h)  # Apply non-linear activation
        z = self.enc_fc(h)
        z = self.enc_bn2(z)
        z = self.dropout(z)
        return z
    
    def decode(self, z, seq_len):
        h = self.dec_fc(z).unsqueeze(0).repeat(seq_len, 1, 1).transpose(0, 1)
        batch_size, seq_len, _ = h.size()
        # import pdb; pdb.set_trace()
        h = h.reshape(batch_size * seq_len, -1)
        h = self.dec_bn1(h)
        h = h.reshape(batch_size, seq_len, -1)
        h = self.dec_nonlinear(h)  # Apply non-linear activation
        output, _ = self.dec_lstm(h)
        output = output.reshape(batch_size * seq_len, -1)
        output = self.dec_bn2(output)
        output = output.reshape(batch_size, seq_len, -1)
        return output

    def forward(self, x):
        seq_len = x.size(1)
        z = self.encode(x)
        out = self.decode(z, seq_len)
        return out



# 定义变分自编码器模型
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 编码器部分
        self.encoder_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器部分
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def encode(self, x):
        _, (h, _) = self.encoder_rnn(x)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        h = self.decoder_input(z).unsqueeze(0).repeat(seq_len, 1, 1).transpose(0, 1)
        output, _ = self.decoder_rnn(h)
        return output

    def forward(self, x):
        
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, seq_len)
        return reconstructed, mu, logvar