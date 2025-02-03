import torch
import numpy as np

class ChannelImportanceAnalyzer:
    def __init__(self, model, input_features, channel_names):
        self.model = model
        self.input_features = input_features
        self.channel_names = channel_names

    def get_top_channels(self, top_n):
        self.model.eval()

        input_features = self.input_features.clone().detach().requires_grad_(True)

        output = self.model(input_features)

        score, _ = output.max(dim=1)

        self.model.zero_grad()
        score.backward(torch.ones_like(score))

        importance_scores = input_features.grad.abs().mean(dim=0).cpu().numpy()

        num_channels = len(self.channel_names)
        features_per_channel = len(importance_scores) // num_channels

        channel_importance = np.array([
            importance_scores[i * features_per_channel:(i + 1) * features_per_channel].mean()
            for i in range(num_channels)
        ])

        top_indices = np.argsort(channel_importance)[-top_n:][::-1] 
        top_channels = [(self.channel_names[i], channel_importance[i]) for i in top_indices]

        return top_channels
