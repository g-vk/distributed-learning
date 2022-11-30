import logging
import numpy as np

import torch

from pathlib import Path
from typing import Dict, Callable

from dc_framework.data_preparation import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def init(model: torch.nn.Module, criterion: torch.nn.Module = None, **kwargs):
    return DCFramework(model, criterion, **kwargs)


class DCFramework:
    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            lr=1e-3,
            epochs=1,
    ):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.criterion = criterion
        self.epochs = epochs
        self.distributed = False

        count_available_gpus = torch.cuda.device_count()
        if count_available_gpus == 0:
            self.device = torch.device("cpu")
        elif count_available_gpus == 1:
            self.device = torch.device("cuda:0")
        else:
            self.distributed = True
            torch.distributed.init_process_group(
                "nccl",
                world_size=count_available_gpus,
            )
            self.local_rank = torch.distributed.get_rank()
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def forward(self, feature, target=None):
        try:
            output = self.model(feature)
        except:
            logger.warning(f"feature: {feature}")
            raise

        loss = None
        if target is not None:
            try:
                loss = self.criterion(output, target)
            except:
                logger.warning(f"output: {output}")
                logger.warning(f"target: {target}")
                raise
        return {
            "output": output,
            "loss": loss
        }

    def train(self, train_data: Dict[str, np.array], validation_data: Dict[str, np.array] = None,  batch_size: int = 1):
        train_data = Dataset(train_data, distributed=self.distributed)
        train_dataloader = train_data.get_dataloader(batch_size=batch_size)

        perform_validation = False
        if validation_data:
            perform_validation = True
            validation_data = Dataset(validation_data, distributed=self.distributed)
            validation_dataloader = validation_data.get_dataloader(batch_size=batch_size)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        for epoch in range(self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Put the model into training mode
            self.model.train()

            # Reset the total loss for this epoch
            total_train_loss = 0
        
            for batch in train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)

                # Clear any previously calculated gradients
                self.model.zero_grad()

                # Perform a forward pass and evaluate the model on this training batch
                output = self.forward(*batch)
                loss = output["loss"]
                # Accumulate the training loss over all of the batches
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients
                loss.backward()

                # Update parameters and take a step using the computed gradient.
                self.optimizer.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # ========================================
            #               Validation
            # ========================================

            if not perform_validation:
                continue

            # Tracking variables
            total_val_loss = 0

            # Put the model in evaluation mode
            self.model.eval()

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                batch = tuple(t.to(self.device) for t in batch)

                # Do not need compute graph during the forward pass for validation
                with torch.no_grad():
                    output = self.forward(*batch)

                    loss = output["loss"]
                    # Accumulate the validation loss over all of the batches
                    total_val_loss += loss.item()

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_val_loss / len(validation_dataloader)

            training_stats.append(
                {
                    'epoch': epoch + 1,
                    'training loss': avg_train_loss,
                    'validation loss': avg_val_loss,
                }
            )

            logger.info(f"{training_stats}")

            return training_stats

    def test(self, test_data: Dict[str, np.array], metric: Callable, batch_size: int = 1):
        test_data = Dataset(test_data)
        test_dataloader = test_data.get_dataloader(batch_size=batch_size)

        # Tracking variables
        predictions, true_labels = [], []

        # Put model in evaluation mode
        self.model.eval()

        # Predict
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            features, labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                output = self.forward(features)

                probs = output["output"]

                # Move probs and labels to CPU
                probs = probs.detach().cpu().numpy()
                labels = labels.to('cpu').numpy()

                predictions.append(probs)
                true_labels.append(labels)

        # Combine the results across all batches.
        flat_predictions = np.concatenate(predictions, axis=0)

        # For each sample, pick the label (0 or 1) with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        # Combine the correct labels for each batch into a single list.
        flat_true_labels = np.concatenate(true_labels, axis=0)

        return metric(flat_predictions, flat_true_labels)

    def save(self, path: Path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: Path):
        state = torch.load(path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

