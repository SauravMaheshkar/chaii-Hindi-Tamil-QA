import json

import torch
import transformers
import wandb

from .utils import AverageMeter, loss_fn, set_seed

__all__ = ["Trainer", "Evaluator"]


class Trainer:
    def __init__(self, model, tokenizer, optimizer, scheduler):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(
        self, config: dict, accelerate_object, train_dataloader, epoch, result_dict
    ):
        count = 0

        # Initialize Accelerator and a AverageMeter
        losses = AverageMeter()

        # -------- Output Prettification ✨ -------- #
        if accelerate_object.is_main_process:
            transformers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()

        self.model.zero_grad()
        self.model.train()

        # Set Random Seed
        set_seed(config["seed"])

        # Prepare everything
        self.model, self.optimizer, train_dataloader = accelerate_object.prepare(
            self.model, self.optimizer, train_dataloader
        )

        for batch_idx, batch_data in enumerate(train_dataloader):
            input_ids, attention_mask, targets_start, targets_end = (
                batch_data["input_ids"],
                batch_data["attention_mask"],
                batch_data["start_position"],
                batch_data["end_position"],
            )

            input_ids, attention_mask, targets_start, targets_end = (
                input_ids.cuda(),
                attention_mask.cuda(),
                targets_start.cuda(),
                targets_end.cuda(),
            )

            outputs_start, outputs_end = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            loss = loss_fn((outputs_start, outputs_end), (targets_start, targets_end))
            loss = loss / config["gradient_accumulation_steps"]

            accelerate_object.backward(loss)

            count += input_ids.size(0)
            wandb.log({"Training Loss": loss.item()})
            losses.update(loss.item(), input_ids.size(0))

            accelerate_object.clip_grad_norm_(
                self.model.parameters(), config["max_grad_norm"]
            )

            if (
                batch_idx % config["gradient_accumulation_steps"] == 0
                or batch_idx == len(train_dataloader) - 1
            ):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            if (batch_idx % config["logging_steps"] == 0) or (batch_idx + 1) == len(
                train_dataloader
            ):
                _s = str(len(str(len(train_dataloader.sampler))))
                ret = [
                    ("Epoch: {:0>2} [{: >" + _s + "}/{} ({: >3.0f}%)]").format(
                        epoch,
                        count,
                        len(train_dataloader.sampler),
                        100 * count / len(train_dataloader.sampler),
                    ),
                    "Train Loss: {: >4.5f}".format(losses.avg),
                ]
                accelerate_object.print(", ".join(ret))

        result_dict["train_loss"].append(losses.avg)
        return result_dict


class Evaluator:
    def __init__(self, model):
        self.model = model

    def save(self, result, output_dir):
        with open(f"{output_dir}/result_dict.json", "w") as f:
            f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))

    def evaluate(self, valid_dataloader, epoch, result_dict):

        losses = AverageMeter()
        for batch_idx, batch_data in enumerate(valid_dataloader):
            # Prepare everything
            self.model = self.model.eval()
            input_ids, attention_mask, targets_start, targets_end = (
                batch_data["input_ids"],
                batch_data["attention_mask"],
                batch_data["start_position"],
                batch_data["end_position"],
            )

            input_ids, attention_mask, targets_start, targets_end = (
                input_ids.cuda(),
                attention_mask.cuda(),
                targets_start.cuda(),
                targets_end.cuda(),
            )

            with torch.no_grad():
                outputs_start, outputs_end = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                loss = loss_fn(
                    (outputs_start, outputs_end), (targets_start, targets_end)
                )
                wandb.log({"Validation Loss": loss.item()})
                losses.update(loss.item(), input_ids.size(0))

        print("----Validation Results Summary----")
        print("Epoch: [{}] Valid Loss: {: >4.5f}".format(epoch, losses.avg))

        result_dict["val_loss"].append(losses.avg)
        return result_dict
