import torch
from custom_ops import Power, Sqrt, Offset

class Model(torch.nn.Module):

    def __init__ (self) -> None:
        super().__init__()
        self.layer1 = Power.forward
        self.layer2 = Sqrt.forward
        self.layer3 = Offset.forward     
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
    
if __name__ == '__main__':
    
    min_value, max_value = 0, 10
    shape = (5, 3, 256, 256)

    random_tensor = torch.randint(min_value, max_value + 1, shape)

    net = Model()
    output = net(random_tensor)

    print(output)