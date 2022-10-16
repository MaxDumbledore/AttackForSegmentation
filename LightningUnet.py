import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from typing import *
from PascalVOC2012 import PascalVOC2012

class LightningUnet(pl.LightningModule):
    def __init__(self, lr=1e-3, **kwargs):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=21,                     # model output channels (number of classes in your dataset)
            activation=None,                # without sigmoid at the end
        )
        params=smp.encoders.get_preprocessing_fn(encoder_name="resnet50", pretrained="imagenet")
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(mode='multiclass')

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

    def shared_step(self, batch: Tuple[torch.Tensor,torch.Tensor], stage):
        image,mask = batch
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multiclass")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(PascalVOC2012(split='train'), batch_size=8, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(PascalVOC2012(split='val'), batch_size=8, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(PascalVOC2012(split='test'), batch_size=8, shuffle=False)



if __name__=='__main__':
    model = LightningUnet()
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model)