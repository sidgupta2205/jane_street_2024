import torch.nn as nn
import torch.nn.functional as F
import torch

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        self.training = False
    def forward(self, x):
        if self.training: 
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=79, encoded_dim=32):
        super(AutoEncoder, self).__init__()
        self.noise =  GaussianNoise(std=0.2)
        # Encoder: Maps input_dim to encoded_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),  # Swish activation
            nn.Dropout(0.2),
            nn.Linear(64, encoded_dim),
            nn.BatchNorm1d(encoded_dim),
            nn.SiLU()  # Swish activation
        )

        # Decoder: Maps encoded_dim back to input_dim
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),  # Swish activation
            nn.Dropout(0.2),
            nn.Linear(64, input_dim) # Use Sigmoid to map output in the range [0, 1], modify if not required
        )

    def forward(self, x,training = True):
        if training:
            x = self.noise(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class MLP(nn.Module):
    def __init__(self, input_dim=79, encoded_dim=32, output_dim=1):
        super(MLP, self).__init__()
        self.noise =  GaussianNoise(std=0.4)
        # MLP: Takes concatenated input_dim + encoded_dim and outputs a single value
        self.encoder_to_mlp = nn.Sequential(
            nn.Linear(input_dim+encoded_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.SiLU(),  # Swish activation
            nn.Dropout(0.2)
        )
        self.mlp = nn.Sequential(
            nn.Linear( input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),  # Swish activation
            nn.Dropout(0.3),
            nn.Linear( 128, 192),
            nn.BatchNorm1d(192),
            nn.Tanh(),  # Swish activation
            nn.Dropout(0.3),
            nn.Linear(192, 16),
            # nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)  # Swish activation
            # nn.Dropout(0.2),
            
        )

    def forward(self, x,training = True,encoder_input = False):
        if training:
            x = self.noise(x)
        if encoder_input:
            x = self.encoder_to_mlp(x)
        return self.mlp(x)*5
    
# Model with combining mlp and autorncoder where the encoder is freezed
class CombinedModel(nn.Module):
    # path for freezed encoder 
    def __init__(self, encoder, mlp):
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.noise =  GaussianNoise(std=0.4)
        self.mlp = mlp
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x,train=True):
        if train:
            x = self.noise(x)
        encoded, _ = self.encoder(x,False)
        return self.mlp(torch.concat([encoded,x],dim=1),False,encoder_input=True)*5