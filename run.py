import os
import pytorch_lightning as pl
from VAE import VAEModel
from dataset import RGBDDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


folder_path = 'xxx'
checkpoint_path = 'xxx'
os.makedirs(checkpoint_path, exist_ok=True)

tb_logger = TensorBoardLogger(save_dir="xxx", name="vae_depth_images")

lr = 1e-4
batch_size = 64
num_episodes = 10 #10% of all data
num_workers = 8

vae = VAEModel(lr)
data_module = RGBDDataModule(folder_path=folder_path, batch_size=batch_size, num_episodes=num_episodes, num_workers=num_workers)
data_module.setup()

epochs = 10

checkpoint_callback = ModelCheckpoint(
    monitor='total_loss',
    filename='model_{epoch:02d}_{total_loss:.4f}',
    dirpath= checkpoint_path,
    save_top_k=1,
    mode='min'
)
if __name__ == "__main__":
    trainer = pl.Trainer(max_epochs=epochs, gpus=1, callbacks=[checkpoint_callback], logger= tb_logger)
    trainer.fit(vae, datamodule=data_module)
