import os
import json
import yaml
import math
import pickle
from datetime import datetime

import wandb
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import _LRScheduler


DEMOGRAPHIC_TEMPLATE = lambda age, gender, education, ethnicity: f'''
You are a {age} years old {ethnicity.lower()} {gender.lower()} whose education level is "{education}".
'''

PERSONALITY_TRAITS_TEMPLATE = lambda openness, conscientiousness, extraversion, agreeableness, emotional_stability: f'''
You are a {openness}{conscientiousness}{extraversion}{agreeableness}{emotional_stability} person.
'''


class FileIO:

    @staticmethod
    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_json(path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_yaml(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)


class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_ratio=0.1, last_epoch=-1):
        """
        Cosine learning rate scheduler with warmup.
        
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_steps (int): Total number of steps for the scheduler.
            warmup_ratio (float): Proportion of steps to spend on warmup. Default is 0.1.
            last_epoch (int): The index of the last epoch. Default is -1.
        """
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.cosine_steps = total_steps - self.warmup_steps
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            # Warmup phase
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_step = step - self.warmup_steps
            return [
                base_lr * 0.5 * (1 + math.cos(math.pi * cosine_step / self.cosine_steps))
                for base_lr in self.base_lrs
            ]


class Trainer:

    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        train_aux_dataloader: DataLoader | None,
        val_dataloader: DataLoader,
        val_aux_dataloader: DataLoader | None,
        batch_size: int, 
        scheduler,
        cur_epoch: int,
        resume_training: int,
        optimizer = None,
        device = "cuda" if torch.cuda.is_available() else "cpu",
        num_epochs = 3,
        output_dir = "outputs",
        project_name = "CogApp Label Smoothing",
    ):

        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.train_aux_dataloader = train_aux_dataloader

        self.val_dataloader = val_dataloader
        self.val_aux_dataloader = val_aux_dataloader

        self.loss_fn = CrossEntropyLoss()
        self.scheduler = scheduler
        self.optimizer = optimizer or AdamW(model.parameters(), lr=5e-5)
        self.gradient_accumulation_steps = 128 // batch_size
        self.cur_epoch = cur_epoch
        self.resume_training = resume_training

        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        wandb.init(project=project_name)

    def train(self):
        best_loss = float('inf')

        running_loss = 0
        # if self.resume_training:
        #     running_loss = pickle.load(open(f'checkpoint-{self.cur_epoch}_loss.pkl', 'rb'))
        
        for epoch in range(self.num_epochs):
            epoch += self.cur_epoch
            # Training
            self.model.train()
            total_loss = 0
            train_steps = 0
            
            if self.train_aux_dataloader:
                progress_bar = tqdm(zip(
                    self.train_dataloader,
                    self.train_aux_dataloader,
                ), desc=f"Epoch {epoch + 1}", total=len(self.train_dataloader))
            else:
                progress_bar = tqdm(
                    self.train_dataloader, 
                    desc=f"Epoch {epoch + 1}", 
                    total=len(self.train_dataloader)
                )

            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(progress_bar):

                if self.train_aux_dataloader:
                    batch, aux_batch = batch
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    aux_batch = {k: v.to(self.device) for k, v in aux_batch.items()}

                    outputs = self.model(
                        context_ids=batch["input_ids"],
                        context_mask=batch["attention_mask"],
                        auxiliary_ids=aux_batch["input_ids"],
                        auxiliary_mask=aux_batch["attention_mask"]
                    )

                    loss = self.loss_fn(outputs, batch["labels"])

                else:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )

                    loss = outputs.loss

                # Backward pass
                loss.backward()
                running_loss += loss.cpu().item()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    wandb.log({"train_loss": avg_running_loss, "learning_rate": cur_lr_rate})

                cur_lr_rate = self.optimizer.param_groups[-1]['lr']
                avg_running_loss = running_loss / (batch_idx + 1)
                
                total_loss += loss.item()
                train_steps += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': total_loss / train_steps})
            
            avg_train_loss = total_loss / train_steps
            wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch + 1})
            
            # Validation
            if self.val_dataloader:
                val_loss = self.evaluate()
                print(f"\nEpoch {epoch + 1}: Train loss = {avg_train_loss:.4f}, Val loss = {val_loss:.4f}")
                wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
                
                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model("best_model", running_loss)
            
            # Save checkpoint
            self.save_model(f"checkpoint-{epoch + 1}", running_loss)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        eval_steps = 0
        
        with torch.no_grad():
            if self.val_aux_dataloader:
                eval_progress_bar = tqdm(zip(
                    self.val_dataloader,
                    self.val_aux_dataloader,
                ), desc="Evaluating", total=len(self.val_dataloader))
            else:
                eval_progress_bar = tqdm(
                    self.val_dataloader, 
                    desc="Evaluating", total=len(self.val_dataloader))

            for batch in eval_progress_bar:
                if self.val_aux_dataloader:
                    batch, aux_batch = batch
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    aux_batch = {k: v.to(self.device) for k, v in aux_batch.items()}

                    outputs = self.model(
                        context_ids=batch["input_ids"],
                        context_mask=batch["attention_mask"],
                        auxiliary_ids=aux_batch["input_ids"],
                        auxiliary_mask=aux_batch["attention_mask"]
                    )

                    loss = self.loss_fn(outputs, batch["labels"])

                else:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs.loss

                total_loss += loss.item()
                eval_steps += 1
        
        return total_loss / eval_steps

    def save_model(self, name, running_loss):
        model_save_path = os.path.join(self.output_dir, f"{name}.pt")
        loss_save_path = os.path.join(self.output_dir, f"{name}_loss.pkl")
        torch.save(self.model.state_dict(), model_save_path)
        pickle.dump(
            running_loss,
            open(loss_save_path, "wb")
        )
        print(f"Model saved to {model_save_path}")


def create_dataloader(data, shuffle, data_collator, batch_size):
    return DataLoader(
        data,
        collate_fn=data_collator,
        batch_size=batch_size,
        shuffle=shuffle
    )


def make_demographic_info(selected_demographic_info: dict) -> str:
    return DEMOGRAPHIC_TEMPLATE(
        age=selected_demographic_info['age'],
        gender=selected_demographic_info['gender'],
        education=selected_demographic_info['education'],
        ethnicity=selected_demographic_info['ethnicity'],
    ).strip()


def make_traits_info(selected_demographic_info: dict) -> str:

    openness_to_experience = '' if \
        selected_demographic_info['open'] == selected_demographic_info['conventional'] else \
            'open, ' if selected_demographic_info['open'] > selected_demographic_info['conventional'] \
            else 'conventional, '

    conscientiousness = '' if \
        selected_demographic_info['dependable'] == selected_demographic_info['disorganized'] else \
            'dependable, ' if selected_demographic_info['dependable'] > selected_demographic_info['disorganized'] \
            else 'disorganized, '

    extraversion = '' if \
        selected_demographic_info['extravert'] == selected_demographic_info['quiet'] else \
            'extravert, ' if selected_demographic_info['extravert'] > selected_demographic_info['quiet'] \
            else 'quiet, '

    agreeableness = '' if \
        selected_demographic_info['sympathetic'] == selected_demographic_info['critical'] else \
            'sympathetic, ' if selected_demographic_info['sympathetic'] > selected_demographic_info['critical'] \
            else 'critical, '

    emotional_stability = '' if \
        selected_demographic_info['anxious'] == selected_demographic_info['calm'] else \
            'anxious, ' if selected_demographic_info['anxious'] > selected_demographic_info['calm'] \
            else 'calm, '

    cur_personality_traits = [
        openness_to_experience,
        conscientiousness,
        extraversion,
        agreeableness,
        emotional_stability,
    ]

    if any([ele != '' for ele in cur_personality_traits]):
        cur_personality_traits = PERSONALITY_TRAITS_TEMPLATE(
            openness=openness_to_experience,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            emotional_stability=emotional_stability,
        )
    else:
        cur_personality_traits = ''

    cur_personality_traits = cur_personality_traits.rsplit(', ', 1)[0]
    if cur_personality_traits.count(', ') >= 1:
        cur_personality_traits, last_trait = cur_personality_traits.rsplit(', ', 1)
        if cur_personality_traits.count(', ') == 1:
            cur_personality_traits += f' and {last_trait} person.'
        else:
            cur_personality_traits += f', and {last_trait} person.'
    else:
        cur_personality_traits += ' person.'

    return cur_personality_traits.strip()