import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset


class CustomDatasetCsv(Dataset):

    def __init__(self, root : str, transform = None) -> None:

        # Memorizzo le trasformazioni che potro' fare sui dati.
        self.transform = transform
                
        self.data_path = Path(root)
        
        # Per prima cosa si controlla il percorso passato in 'root':
        # - Esiste?
        # - E' un file csv?
        # Se ci sono problemi, esco dallo script.
        if not self.__analyze_file():
            sys.exit(-1)
        
        # A questo punto il file e' valida:
        # - Tento di aprirlo come DataFrame pandas.
        # Se ci sono problemi, esco dallo script. 
        if not self.__try_open_as_dataframe():
            sys.exit(-1)

        # A questo punto controllo la struttura del file:
        # - Deve avere due colonne dati, X e Y.
        # - Deve avere almeno un campione, ossia lunghezza non nulla.
        if not self.__check_structure():
            sys.exit(-1)
        
        # Con la certezza che la struttura del file sia corretta, si
        # possono caricare dati, x, ed etichette, y.
        self.__load_data_and_labels()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        
        return self.x_data[index], self.y_labels[index]

    def __analyze_file(self) -> bool:
        
        print(f'Analisi del file dati: {self.data_path.as_posix()}')
        
        if self.data_path.exists():
            if self.data_path.is_dir():
                print(f'{self.data_path.as_posix()} deve essere un file, non una cartella.')
                return False
        else:
            print(f'File {self.data_path.as_posix()} inesistente.')
            return False

        if self.data_path.suffix != '.csv':
            print('Il file deve avere estensione csv.')

        print(f'Il file di dati e\' valido.')
        return True
        
    def __try_open_as_dataframe(self) -> bool:
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f'File aperto correttamente con Pandas.')
            return True
        except:
            print(f'Non e\' stato possibile aprire il file con Pandas.')
            return False        

    def __check_structure(self) -> bool:
        
        # Perche' la struttura sia valida:
        # 1. Devono essere presenti due dimensioni, righe e colonne.
        # 2. Devono essere presenti due colonne dati, tralasciando l'indice.
        # 3. Le colonne devono chiamarsi 'X' e 'Y'.
        # 4. Deve esserci almeno un campione, una riga.
        condition_1 = len(self.df.shape) == 2
        condition_2 = len(self.df.columns) - 1 == 2
        condition_3 = self.df.columns.to_list()[1:] == ['X', 'Y']
        condition_4 = len(self.df) > 0
        
        if condition_1 and condition_2 and condition_3 and condition_4:
            print(f'La struttura del file {self.data_path} e\' valida.')
            return True
        else:
            print(f'La struttura del file {self.data_path} non e\' valida.')
            return False

    def __load_data_and_labels(self) -> None:
        self.x_data = self.df['X'].to_numpy().astype(np.float32)
        self.y_labels = self.df['Y'].to_numpy().astype(np.float32)
        
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

    cdc = CustomDatasetCsv('./test.csv')
    
    for i, data in enumerate(cdc):
        if i == 10:
            break
        print(f'Campione {i}, valore: [{data[0]:.6f}] - etichetta: [{data[1]:.6f}]')

    cdc.show_data()