import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import pytorch_lightning as pl
import os

from Encoder import Encoder
from Decoder import Decoder

class VAEModel(pl.LightningModule):
    def __init__(self, lr, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_kl = 0.0

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sample_latent(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        #BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
        #regularizer = -torch.log(recon_x + 1e-8).mean()
        MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * self.weight_kl
        return MSE, KLD

    def training_step(self, batch, batch_idx):
        #print("train")
        x = batch.reshape(-1, 1, 84, 84)
        #print(x.shape)
        x_recon, mu, logvar = self(x)
        recon_loss, kld_loss = self.loss_function(x_recon, x,  mu, logvar)
        total_loss = recon_loss + kld_loss
        self.log('recon_loss', recon_loss)
        self.log('kld_loss', kld_loss)
        self.log('total_loss', total_loss)

        if self.global_step % 100 == 0:
            # Visualize input images
            self.logger.experiment.add_image('Input Images', x[0], global_step=self.global_step)

            # Visualize reconstructed images
            self.logger.experiment.add_image('Reconstructed Images', x_recon[0], global_step=self.global_step)

        return total_loss

    def validation_step(self, batch, batch_idx):
        #print("val")
        x = batch.reshape(-1, 1, 84, 84)
        #print(x.shape)
        x_recon, mu, logvar = self(x)
        recon_loss, kld_loss = self.loss_function(x_recon, x, mu, logvar)
        total_loss = recon_loss + self.weight_kl * kld_loss
        self.log('recon_loss', recon_loss)
        self.log('kld_loss', kld_loss)
        self.log('total_loss', total_loss)
        return total_loss

    def test_step(self, batch, batch_idx):
        x = batch
        x_recon, mu, logvar = self(x)
        loss = self.loss_function(x_recon, x, mu, logvar)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_end(self):
        self.weight_kl += 0.1
        print("increase kl weigt to", self.weight_kl)
        #self.weight_kl = min(1.0, self.weight_kl)

    def on_train_epoch_start(self):
        print("KL_WEIGHT", self.weight_kl)


    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        model = VAEModel(latent_dim=512)
        return model.load_state_dict(torch.load(checkpoint_path))

