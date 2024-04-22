import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path

np.random.seed(42)

from net import Net


class NetRunner():

    def __init__(self, classes, batch_size, epochs) -> None:
        
        self.classes = classes
        self.batch_size = batch_size
        
        # Creo la rete di classificazione.
        self.net = Net(classes)
        
        self.out_root = Path('./out')
        
        # Il percorso indicato esiste?
        if not self.out_root.exists():
            self.out_root.mkdir()
        
        # Indico dove salvero' il modello addestrato.
        self.outpath_sd = self.out_root / 'trained_model_sd.pth'
        self.outpath = self.out_root / 'trained_model.pth'
        
        # Inizializzo i parametri utili all'addestramento.
        self.lr = 0.001 # tasso di apprendimento.
        self.momentum = 0.9 # momentum.
        self.epochs = epochs # Numero di epoche di addestramento.
        
        # Funzione di costo.
        self.criterion = nn.CrossEntropyLoss()
        
        # Ottimizzatore.
        self.optimizer = optim.SGD(self.net.parameters(), lr = self.lr, momentum = self.momentum)

    def train(self, trainloader : torch.utils.data.DataLoader, valloader : torch.utils.data.DataLoader, preview : bool = False) -> None:

        step_counter = 0 # Conteggio degli step totali.
        step_monitor = 5 # Ogni quanto monitorare la funzione di costo.
        
        ep_monitor = 5

        self.losses_x, self.losses_y = [], []
        self.run_losses_x, self.run_losses_y = [], []

        # Loop di addestramento per n epoche.
        for epoch in range(self.epochs):

            running_loss = 0.0

            # Stop di addestramento. Dimensione batch_size.
            for i, data in enumerate(trainloader, 0):

                # Le rete entra in modalita' addestramento.
                self.net.train()

                # Per ogni input tiene conto della sua etichetta.
                inputs, labels = data
                
                if preview:
                    self.show_preview(inputs, labels)
                    preview = False

                # L'input attraversa al rete. Errori vengono commessi.
                # L'input diventa l'output.
                outputs = self.net(inputs)

                # Calcolo della funzione di costo sulla base di predizioni e previsioni.
                loss = self.criterion(outputs, labels)
                
                # I gradienti vengono azzerati.
                self.optimizer.zero_grad()

                # Avviene il passaggio inverso.
                loss.backward()
                
                # Passo di ottimizzazione
                self.optimizer.step()

                # Monitoraggio statistiche.
                running_loss += loss.item()
                if (i + 1) % step_monitor == 0:
                    self.run_losses_y.append(running_loss / step_monitor)
                    self.run_losses_x.append(step_counter)
                    print(f'GlobalStep: {step_counter:5d} - [Epoca: {epoch + 1:3d}, Step: {i + 1:5d}] loss: {loss.item():.6f} - running_loss: {(running_loss / step_monitor):.6f}')
                    running_loss = 0.0
                
                self.losses_y.append(loss.item())
                self.losses_x.append(step_counter)

                step_counter += 1
                
            if (epoch + 1) % ep_monitor == 0:
                
                val_acc = self.test(valloader, use_current_net=True)
                print(f'GlobalStep: {step_counter:5d} - [Epoca: {epoch + 1:3d}, validation_accuracy: {val_acc:.6f}')
                
        
        print('Finished Training.')

        # Salvataggio del modello e dei parametri
        torch.save(self.net.state_dict(), self.outpath_sd)
        torch.save(self.net, self.outpath)

        print('Model saved.')

        plt.plot(self.losses_x, self.losses_y)
        plt.plot(self.run_losses_x, self.run_losses_y)
        plt.show()

    
    def test(self, testloader : torch.utils.data.DataLoader, use_current_net: bool = False, preview : bool = False):

        total, correct = 0, 0
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        if use_current_net:
            net = self.net
        else:
            net = Net(self.classes)            
            try:
                net.load_state_dict(torch.load(self.outpath_sd))
            except:
                print('Dati modello mancanti.')
                return

        # La rete entra in modalita' inferenza.
        net.eval()

        # Non e' necessario calcolare i gradienti al passaggio dei dati in rete.
        with torch.no_grad():

            # Cicla i campioni di test, batch per volta.
            for data in testloader:

                # Dal batch si estraggono dati ed etichette.
                images, labels = data
                
                if preview:
                    self.show_preview(images, labels)

                # I dati passano nella rete e generano gli output.
                outputs = net(images)

                # Dagli output si evince la predizione finale ottenuta.
                _, predicted = torch.max(outputs.data, 1)

                # Totali e corretti vengono aggiornati.
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

        total_acc = 100 * correct // total
        print(f'Accuratezza totale: {total_acc} %')
        
        print('Accuratezza classi:')
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'|__Classe [{classname:5s}]\t: {accuracy:.2f}%')
            
        return total_acc
            
    
    def denormalize_v1(self, img):
        return np.transpose(img.numpy(), (1, 2, 0))

    def show_preview(self, images, labels):

        cols = 8
        rows = math.ceil(len(images) / cols)

        _, axs = plt.subplots(rows, cols, figsize=(18, 9))
        axs = axs.reshape(rows * cols)
        for ax, im, _ in zip(axs, images, labels):
            ax.imshow(self.denormalize_v1(im))
            #ax.set_title(self.classes[lb.item()])
            ax.grid(False)

        plt.show()