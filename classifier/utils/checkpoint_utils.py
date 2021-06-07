import torch as T

from utils.display_utils import display


def load_temp_checkpoint(agent, time, checkpoint_paths, args, log_paths):
    loaded_stuff = {}

    if args.checkpoint:
        display("Loading checkpoint for the model...\n", log_paths)

        checkpoint = T.load(checkpoint_paths["temp_checkpoint_path"])
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if agent.scheduler is not None:
            agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loaded_stuff["past_epochs"] = checkpoint["past_epochs"]
        loaded_stuff["global_step"] = checkpoint["global_step"]
        loaded_stuff["random_states"] = checkpoint["random_states"]
        loaded_stuff["best_dev_score"] = checkpoint["best_dev_score"]
        loaded_stuff["best_dev_metric"] = checkpoint["best_dev_metric"]
        loaded_stuff["impatience"] = checkpoint["impatience"]
        loaded_stuff["time"] = checkpoint["time"]
        agent.temperature = checkpoint["temperature"]

        display("Restoration Complete\n", log_paths)

    else:
        loaded_stuff["past_epochs"] = 0
        loaded_stuff["global_step"] = 0
        loaded_stuff["random_states"] = None
        loaded_stuff["best_dev_score"] = -float('inf')
        loaded_stuff["best_dev_metric"] = None
        loaded_stuff["impatience"] = 0
        loaded_stuff["time"] = time

    return agent, loaded_stuff


def load_infer_checkpoint(agent, checkpoint_paths, log_paths):
    display("Loading inference weights for the model...\n", log_paths)
    try:
        checkpoint = T.load(checkpoint_paths["infer_checkpoint_path"])
    except:
        checkpoint = T.load(checkpoint_paths["infer_checkpoint_path"], map_location=T.device('cpu'))
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    epochs_taken = checkpoint["epochs"]

    display("Restoration Complete\n", log_paths)

    return agent, epochs_taken


def save_temp_checkpoint(agent, checkpoint_paths, loaded_stuff, log_paths):
    loaded_stuff["model_state_dict"] = agent.model.state_dict()
    loaded_stuff["optimizer_state_dict"] = agent.optimizer.state_dict()
    if agent.scheduler is not None:
        loaded_stuff['scheduler_state_dict'] = agent.scheduler.state_dict()
    loaded_stuff["temperature"] = agent.temperature

    T.save(loaded_stuff, checkpoint_paths["temp_checkpoint_path"])

    display("\nCheckpoint Saved\n\n", log_paths)


def save_infer_checkpoint(epoch, agent, checkpoint_paths, log_paths):
    T.save({
        'model_state_dict': agent.model.state_dict(),
        'epochs': epoch,
    }, checkpoint_paths["infer_checkpoint_path"])

    display("\nInference Weight Saved\n\n", log_paths)
