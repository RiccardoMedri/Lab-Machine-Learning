import math

class Trainer:
    
    def __init__(self):
        self.f_pow = self.pow
        self.f_sqrt = self.sqrt
        self.f_offset = self.offset

    def pow(self, x):
        return x ** 2

    def sqrt(self, x):
        return math.sqrt(x)

    def offset(self, x):
        return x + 100

    def forward(self, x):
        return self.f_offset(self.f_sqrt(self.f_pow(x)))



if __name__ == "__main__":
    trainer = Trainer()
    input = 5
    risultato = trainer.forward(input)
    print("Risultato per input", input, ":", risultato)