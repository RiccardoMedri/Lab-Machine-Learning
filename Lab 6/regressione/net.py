import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_model_summary import summary


class Net(nn.Module):

    def __init__(self):  
        super(Net, self).__init__()

        # Primo strato completamente connesso.
        self.fc1 = nn.Linear(1, 50)

        # Secondo strato completamente connesso.
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x : torch.Tensor) -> torch.Tensor:        
        x = F.relu(self.fc1(x)) # Strato connesso + ReLU.        
        x = self.fc2(x)             # Strato connesso.        
        return x                    # Output.
    
if __name__ == '__main__':

    # Crea l'oggetto che rappresenta la rete.
    n = Net()
    
    # Salva i parametri addestrati della rete.
    torch.save(n.state_dict(), './out/model_state_dict.pth')
    
    # Salva l'intero modello.
    torch.save(n, './out/model.pth')
    
    # Stampa informazioni generali sul modello.
    print(n)

    # Stampa i parametri addestrabili.
    for name, param in n.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # Stampa un recap del modello.
    print(summary(n, torch.ones(size=(40, 1))))