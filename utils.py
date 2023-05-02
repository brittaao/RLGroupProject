import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1']:
        return torch.tensor(obs, device=device).float().unsqueeze(0)
    elif env in ['ALE/Pong-v5']:
        normalize(obs)
        return torch.tensor(obs, device=device).float().unsqueeze(0)
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')


def normalize(data):
    return data / 255