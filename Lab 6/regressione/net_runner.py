import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

np.random.seed(42)

from net import Net


class NetRunner(): 

    def __init__(self, epochs) -> None:
        
        # Creo la rete di classificazione.
        self.net = Net()
        
        self.out_root = Path('./out')
        
        # Il percorso indicato esiste?
        if not self.out_root.exists():
            self.out_root.mkdir()
            
        # Indico dove salvero' il modello addestrato.
        self.outpath_sd = self.out_root / 'trained_model_sd.pth'
        self.outpath = self.out_root / 'trained_model.pth'
            
        # Inizializzo i parametri utili all'addestramento.
        self.lr = 0.01 # tasso di apprendimento.
        self.momentum = 0 # momentum.
        self.epochs = epochs # Numero di epoche di addestramento.
        
        # Funzione di costo.
        self.criterion = nn.MSELoss()
        
        # Ottimizzatore.
        self.optimizer = optim.SGD(self.net.parameters(), lr = self.lr, momentum = self.momentum)

    def train(self, trainloader : torch.utils.data.DataLoader) -> None:

        step_counter = 0 # Conteggio degli step totali.
        step_monitor = 25 # Ogni quanto monitorare la funzione di costo.

        self.losses_x, self.losses_y = [], []
        self.run_losses_x, self.run_losses_y = [], []
        
        # Ciclo di addestramento, epoca per epoca.
        for epoch in range(self.epochs):
            
            running_loss = 0.0

            # Stop di addestramento. Dimensione batch_size.
            for i, data in enumerate(trainloader, 0):   
                
                # Le rete entra in modalita' addestramento.
                self.net.train()
            
                # Per ogni input tiene conto della sua etichetta.
                inputs, labels = data
                
                # L'input attraversa al rete. Errori vengono commessi.
                # L'input diventa l'output.
                outputs = self.net(inputs.unsqueeze(1))

                # Calcolo della funzione di costo sulla base di predizioni e previsioni.
                loss = self.criterion(outputs.squeeze(), labels)

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

        print('Finished Training.')

        # Salvataggio del modello e dei parametri
        torch.save(self.net.state_dict(), self.outpath_sd)
        torch.save(self.net, self.outpath)

        print('Model saved.')

        plt.ylim(0, 1)
        plt.plot(self.losses_x, self.losses_y)
        plt.plot(self.run_losses_x, self.run_losses_y)
        plt.show()
        
    def test(self, testloader : torch.utils.data.DataLoader):

        net = Net()
        
        try:
            net.load_state_dict(torch.load('./out/trained_model_sd.pth'))
        except:
            print('Dati modello mancanti.')
            return

        # La rete entra in modalita' inferenza.
        net.eval()

        # Non e' necessario calcolare i gradienti al passaggio dei dati in rete.
        with torch.no_grad():
            
            data_x = []
            data_y = []
            pred_y = []

            # Cicla i campioni di test, batch per volta.
            for data in testloader:

                # Dal batch si estraggono dati ed etichette.
                data, labels = data

                # I dati passano nella rete e generano gli output.
                outputs = net(data.unsqueeze(1))
                
                data_x += data.tolist()
                data_y += labels.tolist()
                pred_y += outputs.tolist()
            
            data_x = np.array(data_x, dtype=np.float32)
            data_y = np.array(data_y, dtype=np.float32)
            pred_y = np.array(pred_y, dtype=np.float32)
            
            plt.plot(data_x, data_y, label="Actual_test")
            plt.plot(data_x, pred_y, label="Predicted_test")
            plt.legend()
            plt.show()
