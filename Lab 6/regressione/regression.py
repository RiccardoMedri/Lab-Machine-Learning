import torch
import numpy as np

torch.manual_seed(42)

from net_runner import NetRunner
from custom_dataset_csv import CustomDatasetCsv
from custom_dataset_func import CustomDatasetFunc

TRAINING        = True
CSV_DATASET     = True
PREVIEW_DATA    = True
BATCH_SIZE      = 32
SHUFFLE         = True
EPOCHS          = 100

if __name__=="__main__":
    
    # Quali trasformazioni applichiamo ai dati?
    transform = None
    
    # Posso raccogliere i dati tramite due dataset creati ad-hoc.
    # - Uno che se li genera autonomamente e li restituisce.
    # - Uno che li legge da csv.
    tr_dataset = CustomDatasetFunc(min_x=-np.pi, max_x=np.pi, num_data=2000, transform=None)
    tr_dataset.dump_csv('train.csv')
    
    te_dataset = CustomDatasetFunc(min_x=-np.pi, max_x=np.pi, num_data=2000, transform=None)
    te_dataset.dump_csv('test.csv')
    
    if CSV_DATASET:
        tr_dataset = CustomDatasetCsv('./data/train.csv', transform=None)
        te_dataset = CustomDatasetCsv('./data/test.csv', transform=None)
        
    if PREVIEW_DATA and CSV_DATASET:
        tr_dataset.show_data()
        te_dataset.show_data()
        
    # In entrambi i casi passo le trasformazioni da applicare.
    
    # Creo poi i dataloader che prendono i dati dal dataset:
    # - lo fanno a pezzi di dimensione 'batch_size'.
    # - i pezzi li compongono di campioni rando se abilitato 'shuffle'.
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    te_loader = torch.utils.data.DataLoader(te_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Creo l'oggetto che mi permettera' di addestrare e testare il modello.
    runner = NetRunner(EPOCHS)

    # Se abilito 'TRAINING' eseguo addestramento, altrimenti test.
    if TRAINING:
        runner.train(tr_loader)
        runner.test(te_loader)
    else:
        runner.test(te_loader)