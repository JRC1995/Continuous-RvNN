import torch as T
import torch.nn as nn

from controllers.optimizer_controller import get_optimizer


class NLI_agent:
    def __init__(self, model, config, device):
        self.model = model
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = get_optimizer(config)

        if config["optimizer"].lower() == "adam_":
            self.optimizer = optimizer(self.parameters,
                                       lr=config["lr"],
                                       weight_decay=config["weight_decay"],
                                       betas=(0, 0.999),
                                       eps=1e-9)
        else:
            self.optimizer = optimizer(self.parameters,
                                       lr=config["lr"],
                                       weight_decay=config["weight_decay"])

        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                mode='max',
                                                                factor=0.5,
                                                                patience=config["scheduler_patience"])

        self.epoch_level_scheduler = True
        self.config = config
        self.device = device
        self.DataParallel = config["DataParallel"]

        self.temperature = None

        self.optimizer.zero_grad()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def loss_fn(self, logits, labels, train=True, penalty=None):
        loss = self.criterion(logits, labels)
        if penalty is not None and train:
            loss = loss + self.config["penalty_gamma"] * penalty.mean()
        return loss

    def run(self, batch, train=True):

        if train:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        batch["temperature"] = self.temperature

        output_dict = self.model(batch)
        logits = output_dict["logits"]
        labels = batch["labels"].to(logits.device)

        if "penalty" in output_dict:
            penalty = output_dict["penalty"]
        else:
            penalty = None
        loss = self.loss_fn(logits, labels, train, penalty)

        class_probs = T.softmax(logits, dim=-1)
        predictions = T.argmax(logits, dim=-1)

        predictions = predictions.detach().cpu().numpy().tolist()
        class_probs = class_probs.detach().cpu().numpy().tolist()

        labels = batch["labels"].numpy().tolist()

        predicted_ground_truth_probs = []
        for label, class_prob in zip(labels, class_probs):
            predicted_ground_truth_probs.append(class_prob[label])

        metrics = self.eval_fn(predictions, labels)
        metrics["loss"] = loss.item()

        items = {"display_items": {"sequences1": batch["sequences1"],
                                   "sequences2": batch["sequences2"],
                                   "predictions": predictions,
                                   "pairIDs": batch["pairIDs"],
                                   "labels": labels},
                 "loss": loss,
                 "metrics": metrics,
                 "stats_metrics": {"predicted_ground_truth_probs": predicted_ground_truth_probs}}

        return items

    def backward(self, loss):
        loss.backward()

    def step(self):
        if self.config["max_grad_norm"] is not None:
            T.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.config["current_lr"] = self.optimizer.param_groups[-1]["lr"]

    def eval_fn(self, predictions, labels):

        correct_predictions = 0
        total = 0
        for prediction, label in zip(predictions, labels):
            if prediction == label:
                correct_predictions += 1
            total += 1

        accuracy = correct_predictions / total if total > 0 else 0
        accuracy *= 100

        return {"correct_predictions": correct_predictions,
                "total": total,
                "accuracy": accuracy}
