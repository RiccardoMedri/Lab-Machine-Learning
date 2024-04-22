import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset


class CustomDatasetFunc(Dataset):

    def __init__(self, min_x, max_x, num_data, transform = None) -> None:

        # Memorizzo le trasformazioni che potro' fare sui dati.
        self.transform = transform
                
        self.x_data = np.linspace(min_x, max_x, num_data).astype(np.float32)
        self.y_labels = np.sin(self.x_data).astype(np.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        
        return self.x_data[index], self.y_labels[index]
    
    def dump_csv(self, filename : str):
        
        out_path = Path('./data/')
        
        if not out_path.exists():
            out_path.mkdir()
            
        filename = out_path / filename
        
        df = pd.DataFrame({'X': self.x_data, 'Y': self.y_labels})
        df.index.name = 'Id'
        df.to_csv(filename, index=True)

    def show_data(self, data_start = None, data_end = None):
        
        data_start = 0 if data_start is None else data_start
        data_end = len(self.df) if data_end is None else data_end
        
        s = min(data_start, data_end)
        e = max(data_start, data_end)
        
        s = s if (s >= 0 and s <= len(self.df)) else 0
        e = e if (e >= 0 and e <= len(self.df)) else 0
        
        self.df['Y'][s:e].plot()
        plt.show()

if __name__ == '__main__':

    cdc = CustomDatasetFunc(min_x=-np.pi, max_x=np.pi, num_data=2000)
    
    for i, data in enumerate(cdc):
        if i == 10:
            break
        print(f'Campione {i}, valore: [{data[0]:.6f}] - etichetta: [{data[1]:.6f}]')

    cdc.show_data()