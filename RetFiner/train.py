import torch

from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics
import argparse

from util_funcs import get_model_name, get_data_loaders
from RFM import RFM



parser = argparse.ArgumentParser()
parser.add_argument('--contra_temp', type=float, default=0.07)
parser.add_argument('--exponential_temp', type=bool, default=False)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--losses', nargs='+', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--note', type=str, default='')
parser.add_argument('--model_weights', type=str, default='')

args, unknown = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, val_loader = get_data_loaders(args.model_weights.lower(), args)
viz_backbone_name = get_model_name(args.model_weights.lower())
model = RFM(vision_weights_path=args.model_weights)

model_save = './models/' + viz_backbone_name + str(args.contra_temp) + str(args.exponential_temp) + 'lr' + str(args.lr) + 'batch_size' + str(args.batch_size) + 'LossesAll' + args.note + '/'

best_model_checkpoint = ModelCheckpoint(
    dirpath=model_save,               # Directory to save checkpoints
    filename="best-model",            # Filename for the best model
    save_top_k=1,                     # Keep only the best model
    monitor="val_epoch_loss",               # Monitor validation loss
    mode="min",                       # Save when 'val_loss' is minimum
    save_last=False,                  # Don't save a separate last checkpoint
    save_on_train_epoch_end=True      # Save at the end of each validation (epoch end)
)

step_checkpoint = ModelCheckpoint(
    dirpath=model_save,               # Directory to save checkpoints
    filename="step-{step:06d}",       # Filename with step number
    save_top_k=1,                     # Keep only the latest step checkpoint
    every_n_train_steps=1000,           # Save every n steps
    save_on_train_epoch_end=False,    # Save based on step intervals
    save_last=False                   # Don't save a separate last checkpoint
)

early_stopping = EarlyStopping(
    monitor='val_epoch_loss',  # The metric to monitor
    patience=3,                # Number of epochs with no improvement after which training will be stopped
    mode='min',                # Stop when the monitored quantity stops decreasing (minimization)
    verbose=True               # Whether to log the early stopping information
)

callbacks = [best_model_checkpoint, step_checkpoint, early_stopping]


class LightningRetFiner(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_metrics = {
            'train': torchmetrics.MeanMetric(),
            'val': torchmetrics.MeanMetric(),
            'test': torchmetrics.MeanMetric(),
        }

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx, prefix):
        loss = self(batch)#.loss#self.model(batch)#, rank=self.global_rank)
        self.loss_metrics[prefix].cpu().update(loss.detach().cpu())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'train')
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'val')
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, 'test')
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=args.lr)

    def compute_loss(self, y_hat, y):
        # Assuming a generic loss function; replace with your specific loss computation
        return torch.nn.functional.cross_entropy(y_hat, y)

    def on_train_epoch_end(self):
        self.log("train/epoch_loss", self.loss_metrics['train'].to(device).compute())
        self.loss_metrics['train'].to(device).reset()

    def on_validation_epoch_end(self):
        epoch_loss = self.loss_metrics['val'].to(device).compute()
        self.log("val/epoch_loss", epoch_loss)
        self.log("val_epoch_loss", epoch_loss)
        self.loss_metrics['val'].to(device).reset()

    def on_test_epoch_end(self):
        epoch_loss = self.loss_metrics['test'].to(device).compute()
        self.log("test/epoch_loss", epoch_loss)
        self.loss_metrics['test'].to(device).reset()


lit_model = LightningRetFiner(model).to(device)

trainer = Trainer(
    accelerator="gpu",
    precision=16,
    num_sanity_val_steps=0,
    callbacks=callbacks,
    max_epochs=20
)

lit_model.train()

trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
