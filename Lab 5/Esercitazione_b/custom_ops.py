import torch

class Power(torch.nn.module):

    def __init__(self):
        super().__init__()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x ** 2
    

class Sqrt(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)

    
class Offset(torch.nn.Module):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x + self.value