import torch
import yaml


class Utils:
    @staticmethod
    def load_config_yaml(file_path):
        with open(file_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_actor(file_path, actor_class, num_bees):
        actor = actor_class(num_bees)
        if torch.cuda.is_available():
            actor.load_state_dict(torch.load(file_path))
        else:
            actor.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")))
        return actor

    @staticmethod
    def load_critic(file_path, critic_class, global_state_size, num_bees):
        critic = critic_class(global_state_size, num_bees)
        if torch.cuda.is_available():
            critic.load_state_dict(torch.load(file_path))
        else:
            critic.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")))
        return critic
