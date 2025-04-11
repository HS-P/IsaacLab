import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvEncoderMLP(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=8):
        super(EnvEncoderMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.fc3(x)
        return z

# 예시 사용법
if __name__ == '__main__':
    # 배치 사이즈 16로 9차원 입력 생성
    sample_input = torch.randn(16, 9)
    encoder = EnvEncoderMLP(input_dim=9, hidden_dim=64, output_dim=8)
    z = encoder(sample_input)
    print("Extracted latent vector shape:", z.shape)
    # 출력: Extracted latent vector shape: torch.Size([16, 8])