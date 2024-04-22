import torch
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(42)

from net_runner import NetRunner
from custom_dataset_shapes import CustomDatasetShapes

TRAINING        = True
CUSTOM_DATASET  = False
PREVIEW_DATA    = False
BATCH_SIZE      = 24
SHUFFLE         = True
EPOCHS          = 25

if __name__ == "__main__":

    # Quali trasformazioni applichiamo alle immagini?
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Posso raccogliere le immagini con il dataset custom creato appositamente.
    # - Posso farlo sia per i dati di training.
    # - Che per quelli di test e/o validazione.
    tr_dataset = CustomDatasetShapes(root='../generatore_forme/dst/training', transform=transform)
    va_dataset = CustomDatasetShapes(root='../generatore_forme/dst/validation', transform=transform)
    te_dataset = CustomDatasetShapes(root='../generatore_forme/dst/test', transform=transform)
    classes = tr_dataset.classes

    # Se non voglio usare il dataset custom, posso usarne uno di base fornito da PyTorch.
    # Questo rappresenta genericamente:
    # - Un dataset di immagini.
    # - Diviso in sotto-cartelle.
    # - Il nome delle sotto-cartelle rappresenta il nome della classe.
    # - In ogni sotto-cartella ci sono solo immagini di quella classe.
    if not CUSTOM_DATASET:
        tr_dataset = torchvision.datasets.ImageFolder(root='../generatore_forme/dst/training', transform=transform)
        va_dataset = torchvision.datasets.ImageFolder(root='../generatore_forme/dst/validation', transform=transform)
        te_dataset = torchvision.datasets.ImageFolder(root='../generatore_forme/dst/test', transform=transform)

    # In entrambi i casi passo le trasformazioni da applicare.

    # Creo poi i dataloader che prendono i dati dal dataset:
    # - lo fanno a pezzi di dimensione 'batch_size'.
    # - i pezzi li compongono di campioni rando se abilitato 'shuffle'.
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    va_loader = torch.utils.data.DataLoader(va_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    te_loader = torch.utils.data.DataLoader(te_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    # Creo l'oggetto che mi permettera' di addestrare e testare il modello.
    runner = NetRunner(classes, BATCH_SIZE, EPOCHS)

    # Se abilito 'TRAINING' eseguo addestramento, altrimenti test.
    if TRAINING:
        runner.train(tr_loader, va_loader, PREVIEW_DATA)
        runner.test(te_loader, preview = PREVIEW_DATA)
    else:
        runner.test(te_loader, preview = PREVIEW_DATA)