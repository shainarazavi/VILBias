import argparse
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from datasets import load_dataset
from lightning import seed_everything
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer import Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from timm.data.transforms import ResizeKeepRatio
from transformers import (
    AutoModelForTextEncoding,
    AutoTokenizer,
    CLIPVisionModel,
    PreTrainedTokenizerBase,
)
from typing_extensions import Literal

import wandb

wandb.require("core")
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val", "test"],
        text_tokenizer: PreTrainedTokenizerBase,
        image_size: int = 224,
        adf: pd.DataFrame = None,
    ) -> None:
        assert os.path.exists(root_dir), f"Directory {root_dir} does not exist"
        assert split in ["train", "val", "test"], f"Invalid split {split}"

        self.root_dir = root_dir

        if adf is not None and isinstance(adf, pd.DataFrame):
            df = adf
        else:
            df = pd.read_csv(os.path.join(root_dir, "annotations", f"{split}.csv"))

        # label encoding
        label_encoder = LabelEncoder()
        df["combined_label"] = label_encoder.fit_transform(df["combined_label"])
        df["image_label"] = label_encoder.fit_transform(df["image_label"])
        df["text_label"] = label_encoder.fit_transform(df["text_label"])
        df = df[
            [
                "unique_id",
                "text_content",
                "image_filename",
                "image_label",
                "text_label",
                "combined_label",
            ]
        ]
        self._targets = df["combined_label"].values
        self.data = df.to_dict(orient="records")
        self.text_tokenizer = text_tokenizer
        self.img_transform = T.Compose(
            [
                ResizeKeepRatio(512, interpolation="bicubic"),
                T.CenterCrop(image_size),
                T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])
                if split == "train"
                else T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.data[idx]

        with Image.open(
            os.path.join(self.root_dir, "images", sample["image_filename"]), "r"
        ) as img:
            img = self.img_transform(img.convert("RGB"))

        text_encoding = self.text_tokenizer(
            sample["text_content"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": img,
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "image_label": torch.tensor(sample["image_label"], dtype=torch.long),
            "text_label": torch.tensor(sample["text_label"], dtype=torch.long),
            "combined_label": torch.tensor(sample["combined_label"], dtype=torch.long),
        }


class MultimodalClassifier(LightningModule):
    def __init__(
        self,
        text_encoder_id_or_path: str,
        image_encoder_id_or_path: str,
        projection_dim: int,
        fusion_method: Literal["concat", "align"] = "align",
        text_encoder_lora: bool = False,
        image_encoder_lora: bool = False,
        proj_dropout: float = 0.1,
        fusion_dropout: float = 0.1,
        num_classes: int = 1,
        freeze_text_encoder: bool = False,
        freeze_image_encoder: bool = False,
        lr: float = 1e-4,
        wd: float = 1e-4,
        train_labels: list[int] = None,
    ) -> None:
        super().__init__()
        assert fusion_method in [
            "concat",
            "align",
        ], f"Invalid fusion method {fusion_method}"
        self.fusion_method = fusion_method
        self.projection_dim = projection_dim
        self.num_classes = num_classes

        self.text_encoder = AutoModelForTextEncoding.from_pretrained(
            text_encoder_id_or_path
        )
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        elif text_encoder_lora:
            lora_config_text = LoraConfig(r=32, target_modules=["query", "value"])
            self.text_encoder = prepare_model_for_kbit_training(self.text_encoder)
            self.text_encoder = get_peft_model(self.text_encoder, lora_config_text)

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, self.projection_dim),
            nn.Dropout(proj_dropout),
        )

        self.image_encoder = CLIPVisionModel.from_pretrained(image_encoder_id_or_path)
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        elif image_encoder_lora:
            lora_config_image = LoraConfig(r=32, target_modules=["q_proj", "v_proj"])
            self.image_encoder = prepare_model_for_kbit_training(self.image_encoder)
            self.image_encoder = get_peft_model(self.image_encoder, lora_config_image)

        self.image_projection = nn.Sequential(
            nn.Linear(self.image_encoder.config.hidden_size, self.projection_dim),
            nn.Dropout(proj_dropout),
        )

        fusion_input_dim = (
            self.projection_dim * 2
            if fusion_method == "concat"
            else self.projection_dim
        )
        self.fusion_layer = nn.Sequential(
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_input_dim, self.projection_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

        self.classifier = nn.Linear(self.projection_dim, self.num_classes)
        self.lr = lr
        self.wd = wd

        class_weights = None
        if train_labels is not None:
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(train_labels), y=train_labels
            )
            class_weights = torch.tensor(class_weights, device=self.device)

        if self.num_classes == 1:
            pos_weight = None
            if class_weights is not None:
                pos_weight = class_weights[1]
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self._preds = []
        self._targets = []

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_features = self.text_encoder(
            input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)

        image_features = self.image_encoder(
            pixel_values, return_dict=True
        ).pooler_output
        image_features = self.image_projection(image_features)

        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

        if self.fusion_method == "concat":
            fused_features = torch.cat([text_features, image_features], dim=-1)
        elif self.fusion_method == "align":
            fused_features = torch.mul(text_features, image_features)

        fused_features = self.fusion_layer(fused_features)
        return self.classifier(fused_features)

    def _shared_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        labels = batch["combined_label"]

        if self.num_classes == 1:  # binary classification
            logits = logits[:, 0]  # only take the logits for the positive class
            labels = labels.float()

        loss = self.loss_fn(logits.float(), labels)

        return logits, labels, loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        _, _, loss = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        logits, label, loss = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True)

        self._preds.append(self.all_gather(logits))
        self._targets.append(self.all_gather(label))
        return loss

    def on_validation_epoch_end(self):
        predictions = torch.cat(self._preds, dim=0)
        predictions = (predictions.sigmoid() > 0.5).cpu().numpy().astype(int)
        targets = torch.cat(self._targets, dim=0).cpu().numpy()

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            targets, predictions, average="macro"
        )
        report = classification_report(
            targets, predictions, target_names=["Biased", "Unbiased"]
        )
        self.print("Classification Report:\n", report)
        self.log_dict(
            {
                "val/precision": precision,
                "val/recall": recall,
                "val/f1_score": f1_score,
            },
            prog_bar=True,
        )
        self._preds = []
        self._targets = []

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits, label, loss = self._shared_step(batch)
        self.log("test/loss", loss, prog_bar=True)

        self._preds.append(self.all_gather(logits))
        self._targets.append(self.all_gather(label))
        return loss

    def on_test_epoch_end(self):
        predictions = torch.cat(self._preds, dim=0)
        predictions = (predictions.sigmoid() > 0.5).cpu().numpy().astype(int)
        targets = torch.cat(self._targets, dim=0).cpu().numpy()

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            targets, predictions, average="macro"
        )
        report = classification_report(
            targets, predictions, target_names=["Biased", "Unbiased"]
        )
        self.print("Classification Report:\n", report)
        self.log_dict(
            {
                "test/precision": precision,
                "test/recall": recall,
                "test/f1_score": f1_score,
            },
            prog_bar=True,
        )
        self._preds = []
        self._targets = []

    def configure_optimizers(self):
        params = [
            {"params": [p for _, p in self.named_parameters() if p.requires_grad]}
        ]
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.wd, eps=1e-6
        )
        warmup_steps = int(self.trainer.estimated_stepping_batches * 0.1)
        assert warmup_steps > 0, "Warmup steps must be greater than 0"

        # cosine scheduler with warmup
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    self.trainer.estimated_stepping_batches - warmup_steps,
                    eta_min=torch.finfo(torch.float32).eps,
                ),
            ],
            milestones=[warmup_steps],
            verbose=True,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def main(args):
    seed_everything(args.seed, workers=True)

    ds = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(args.data_root_dir, "annotations", "train.csv"),
            "val": os.path.join(args.data_root_dir, "annotations", "val.csv"),
            "test": os.path.join(args.data_root_dir, "annotations", "test.csv"),
        },
    )

    # setup dataloaders
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer_path)
    train_dataset = Dataset(
        args.data_root_dir,
        split="train",
        text_tokenizer=text_tokenizer,
        image_size=args.image_size,
        adf=ds["train"].to_pandas(),
    )
    val_dataset = Dataset(
        args.data_root_dir,
        split="val",
        text_tokenizer=text_tokenizer,
        image_size=args.image_size,
        adf=ds["val"].to_pandas(),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # setup model
    model = MultimodalClassifier(
        text_encoder_id_or_path=args.text_encoder_path,
        image_encoder_id_or_path=args.image_encoder_path,
        projection_dim=args.projection_dim,
        proj_dropout=args.proj_dropout,
        fusion_method=args.fusion_method,
        fusion_dropout=args.fusion_dropout,
        freeze_text_encoder=args.freeze_text_encoder,
        freeze_image_encoder=args.freeze_image_encoder,
        lr=args.lr,
        wd=args.wd,
        text_encoder_lora=args.text_encoder_lora,
        image_encoder_lora=args.image_encoder_lora,
        train_labels=train_dataset._targets,
    )

    wandb_logger = WandbLogger(
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        project="nmb-plus",
        entity="vector-institute-aieng",
        resume="allow",
        tags=["multimodal-classification", args.fusion_method],
        config=wandb.helper.parse_config(
            vars(args), exclude=("wandb_run_id", "wandb_run_name")
        ),
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        deterministic=True,
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                dirpath=os.path.join(args.ckpt_dir, args.wandb_run_name),
            ),
            EarlyStopping(monitor="val/loss", mode="min", patience=5),
            LearningRateMonitor(),
        ],
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    if args.run_test_after_training:
        test_dataset = Dataset(
            args.data_root_dir,
            split="test",
            text_tokenizer=text_tokenizer,
            image_size=args.image_size,
            adf=ds["test"].to_pandas(),
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        trainer.test(dataloaders=test_dataloader, verbose=True)


if __name__ == "__main__":
    default_ckpt_dir = (
        f"/checkpoint/{os.getenv('USER')}/{os.getenv('SLURM_JOB_ID')}"
        if os.getenv("SLURM_JOB_ID") is not None
        else "./checkpoints"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_root_dir", type=str, default="/projects/NMB-Plus/consolidated_data"
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--text_tokenizer_path", type=str, default="distilbert/distilroberta-base"
    )
    parser.add_argument(
        "--text_encoder_path", type=str, default="distilbert/distilroberta-base"
    )
    parser.add_argument(
        "--image_encoder_path", type=str, default="openai/clip-vit-large-patch14"
    )
    parser.add_argument("--projection_dim", type=int, default=768)
    parser.add_argument("--proj_dropout", type=float, default=0.1)
    parser.add_argument("--fusion_method", choices=["concat", "align"], default="align")
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--text_encoder_lora", action="store_true", default=False)
    parser.add_argument("--image_encoder_lora", action="store_true", default=False)
    parser.add_argument("--freeze_text_encoder", action="store_true", default=False)
    parser.add_argument("--freeze_image_encoder", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--ckpt_dir", type=str, default=default_ckpt_dir)
    parser.add_argument("--wandb_run_name", type=str, required=True)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--run_test_after_training", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
