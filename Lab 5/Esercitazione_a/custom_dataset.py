import random
import utilities as u

from torch.utils.data import Dataset

TOT_N = 10
MIN_N = 5
MAX_N = 9

class RandomNumberDataset(Dataset):
    
    def __init__(self, transform = None) -> None:
        self.transform = transform
        self.list = [random.uniform(MIN_N, MAX_N) for x in range (TOT_N)]
    
    def __len__(self) -> int:
        return len(self.list)
    
    def __getitem__ (self, index: int) -> float:
        if self.transform is None:
            return self.list[index]
        else:
            return self.transform(self.list[index])
    

if __name__ == '__main__':

    rnd = RandomNumberDataset(transform=u.NUMBER_TRANSFORMATION)

    for i, sample in enumerate(rnd):
        print(f'Sample {i:03}: --> type: {type(sample)} - value: {sample:.3f}')

    rnd_2 = RandomNumberDataset(transform=None)

    for i, sample in enumerate(rnd_2):
        print(f'Sample {i:03}: --> type: {type(sample)} - value: {sample:.3f}')