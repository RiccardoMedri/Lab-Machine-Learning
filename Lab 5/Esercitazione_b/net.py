import torch
from custom_ops import Power, Sqrt, Offset

class Model(torch.nn.Module):

    def __init__ (self) -> None:
        super().__init__()
        self.layer1 = Power()
        self.layer2 = Sqrt()
        self.layer3 = Offset(100)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        result = self.layer1.forward(x)
        result = self.layer2.forward(result)
        result = self.layer3.forward(result)
        return result
    
if __name__ == '__main__':
    
    min_value, max_value = 0, 10
    shape = (5, 3, 256, 256)

    random_tensor = torch.randint(min_value, max_value + 1, shape)

    net = Model()
    output = net(random_tensor)

    print(output)