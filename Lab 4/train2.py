import numpy as np

class Trainer:
    
    def __init__(self):
        self.f_pow = self.pow
        self.f_sqrt = self.sqrt
        self.f_offset = self.offset

    def pow(self, x):
        return np.power(x, 2)

    def sqrt(self, x):
        return np.sqrt(x)

    def offset(self, x):
        return x + 100

    def forward(self, x):
        return self.f_offset(self.f_sqrt(self.f_pow(x)))
    

if __name__ == "__main__":
    trainer = Trainer()
    x = np.random.rand(5, 256, 256, 3)
    risultato1 = trainer.forward(x)
    print("Output dopo l'applicazione delle funzioni:", risultato1)
    lista_array = [np.random.rand(256, 256, 3) for _ in range(5)]
    array_numpy = np.array(lista_array)
    risultato2 = trainer.forward(lista_array)
    print("Output dopo l'applicazione delle funzioni:", risultato2)