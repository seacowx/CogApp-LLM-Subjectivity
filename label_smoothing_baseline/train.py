import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW

from transformers import (
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding
)
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data import CogAppDataLoader
from dual_encoder import DualEncoder
from label_smoother import LabelSmoother
from utils import FileIO, Trainer, create_dataloader, CosineWarmupScheduler

os.environ["WANDB_PROJECT"]="cogg_app_label_smoothing_RoBERTa"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='microsoft/deberta-v3-large',
    )
    parser.add_argument(
        '--modal',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--with_demo', 
        action='store_true', 
        help='Use demographic info',
    )
    parser.add_argument(
        '--with_traits', 
        action='store_true', 
        help='Use personality traits info',
    )
    parser.add_argument(
        '--state_dict_path', 
        type=str, 
        help='the path to the folder for saving model weights',
    )
    parser.add_argument(
        '--resume_state_dict_path', 
        type=str, 
        help='restart training from a checkpoint',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.with_demo or args.with_traits:
        model = DualEncoder()
        dual_encoder = True
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=5)
        dual_encoder = False

    MAX_LENGTH = 512
    BATCH_SIZE = 8

    dataloader = CogAppDataLoader(
        tokenizer=tokenizer,
    )

    train_data, train_demo_data, train_traits_data, train_demo_traits_data, train_data_len = dataloader.load_data(
        data_path='../data/envent_train_unique_hf_processed.json',
        modal=args.modal,
    )
    val_data, val_demo_data, val_traits_data, val_demo_traits_data, val_data_len = dataloader.load_data(
        data_path='../data/envent_valid_unique_hf_processed.json',
        modal=args.modal,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors='pt',
    )

    train_dataloader = create_dataloader(
        data=train_data, 
        shuffle=True,
        data_collator=data_collator,
        batch_size=BATCH_SIZE,
    )
    val_dataloader = create_dataloader(
        data=val_data, 
        shuffle=False,
        data_collator=data_collator,
        batch_size=BATCH_SIZE,
    )

    print(f'Train data length: {train_data_len}')
    print(f'Val data length: {val_data_len}')

    # load auxiliary information 
    train_aux_dataloader = None
    val_aux_dataloader = None
    if args.with_demo and args.with_traits:
        train_aux_dataloader = create_dataloader(
            data=train_demo_traits_data, 
            shuffle=True,
            data_collator=data_collator,
            batch_size=BATCH_SIZE,
        )
        val_aux_dataloader = create_dataloader(
            data=val_demo_traits_data, 
            shuffle=False,
            data_collator=data_collator,
            batch_size=BATCH_SIZE,
        )
    elif args.with_demo:
        train_aux_dataloader = create_dataloader(
            data=train_demo_data, 
            shuffle=True,
            data_collator=data_collator,
            batch_size=BATCH_SIZE,
        )
        val_aux_dataloader = create_dataloader(
            data=val_demo_data, 
            shuffle=False,
            data_collator=data_collator,
            batch_size=BATCH_SIZE,
        )
    elif args.with_traits:
        train_aux_dataloader = create_dataloader(
            data=train_traits_data, 
            shuffle=True,
            data_collator=data_collator,
            batch_size=BATCH_SIZE,
        )
        val_aux_dataloader = create_dataloader(
            data=val_traits_data, 
            shuffle=False,
            data_collator=data_collator,
            batch_size=BATCH_SIZE,
        )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        cur_mse = mean_squared_error(labels, logits)
        cur_mae = mean_absolute_error(labels, logits)
        return {'mse': cur_mse, 'mae': cur_mae}

    modal_name = 'unimodal' if args.modal == 1 else 'bimodal'

    postfix = '' 
    postfix += '_demo' if args.with_demo else ''
    postfix += '_traits' if args.with_traits else ''

    # resume training from a checkpoint
    resume_training = cur_epoch = 0
    if args.resume_state_dict_path:
        print(f"Resuming training from epoch {args.resume_state_dict_path.split('_')[-1].split('.')[0]}")
        model.load_state_dict(torch.load(args.resume_state_dict_path))
        cur_epoch = int(args.resume_state_dict_path.split('-')[-1].split('.')[0])
        resume_training = cur_epoch

    EPOCHS = 30 - cur_epoch
    WARMUP_RATIO = 0.05
    gradient_accumulation_steps = 128 // BATCH_SIZE
    train_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    print(f'Total training steps: {train_steps_per_epoch} per epoch.')
    total_training_steps = train_steps_per_epoch * EPOCHS
    optimizer = AdamW(model.parameters(), lr=2e-7)
    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        total_steps=total_training_steps, 
        warmup_ratio=WARMUP_RATIO,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        train_aux_dataloader=train_aux_dataloader,
        val_dataloader=val_dataloader,
        val_aux_dataloader=val_aux_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=EPOCHS,
        cur_epoch=cur_epoch,
        output_dir=args.state_dict_path,
        batch_size=BATCH_SIZE,
        resume_training=resume_training,
    )

    trainer.train()


if __name__ == '__main__':
    main()
