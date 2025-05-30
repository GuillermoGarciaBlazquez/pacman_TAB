import os
import glob
import json
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from seed import PACMAN_SEED
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Fijamos todas las semillas para reproducibilidad
torch.manual_seed(PACMAN_SEED)
random.seed(PACMAN_SEED)
# Constantes
INPUT_SIZE = None  # Se determinará en tiempo de ejecución basado en el tamaño del mapa
HIDDEN_SIZE = 128
NUM_ACTIONS = 5  # Stop, North, South, East, West
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
MODELS_DIR = "models"

# Mapeo de acciones a índices
ACTION_TO_IDX = {
    'Stop': 0,
    'North': 1,
    'South': 2, 
    'East': 3,
    'West': 4
}

# Mapeo de índices a acciones
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
# Esto es obligatorio para poder usar dataloaders en pytorch
class PacmanDataset(Dataset):
    def __init__(self, maps, actions):
        self.maps = maps
        self.actions = actions
    
    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        map_tensor = torch.FloatTensor(self.maps[idx])
        action_tensor = torch.LongTensor([self.actions[idx]])
        return map_tensor, action_tensor.squeeze()

class PacmanNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PacmanNet, self).__init__()

        self.input_features = input_size[0] * input_size[1]

        self.model = nn.Sequential(
            nn.Linear(self.input_features, hidden_size * 4),
            nn.BatchNorm1d(hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Преобразуем карту в вектор
        return self.model(x)


def load_and_merge_data(data_dir="pacman_data"):
    """Carga todos los archivos CSV de partidas y los combina en un único DataFrame"""
    all_maps = []
    all_actions = []
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Archivos CSV encontrados: {csv_files}")
    
    if not csv_files:
        raise ValueError(f"No se encontraron archivos CSV en {data_dir}")
    
    print(f"Cargando {len(csv_files)} archivos de partidas...")
    
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Solo usar movimientos de Pacman (agente 0)
                if int(row.get('agent_index', 0)) == 0:
                    action = row.get('action')
                    map_matrix = json.loads(row.get('map_matrix', '[]'))
                    
                    # Verificar que los datos sean válidos
                    if action in ACTION_TO_IDX and map_matrix:
                        all_maps.append(map_matrix)
                        all_actions.append(ACTION_TO_IDX[action])
    
    print(f"Datos cargados: {len(all_maps)} ejemplos")
    return all_maps, all_actions

def preprocess_maps(maps):
    """Preprocesa las matrices del juego para preparar los datos de entrada para la red"""
    # Determinar las dimensiones del mapa
    height = len(maps[0])
    width = len(maps[0][0])
    
    # Convertir a numpy array
    processed_maps = np.array(maps).astype(np.float32)
    
    # Normalizar los valores: dividir por 5 (el valor máximo) para obtener valores entre 0 y 1
    processed_maps = processed_maps / 5.0
    
    print(f"Forma de los datos de entrada: {processed_maps.shape}")
    print(f"Tamaño del mapa: {height}x{width}")
    
    return processed_maps, (height, width)


def train_model(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS):
    """
    Train the model and print train/val loss and accuracy for each epoch (not per batch),
    with summary for overfitting/underfitting analysis.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_accuracy = 0.0
    best_model_state = None

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"Comenzando entrenamiento por {num_epochs} épocas...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for maps, actions in train_loader:
            maps, actions = maps.to(device), actions.to(device)
            outputs = model(maps)
            loss = criterion(outputs, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * actions.size(0)
            _, predicted = outputs.max(1)
            train_total += actions.size(0)
            train_correct += predicted.eq(actions).sum().item()
        
        avg_train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for maps, actions in test_loader:
                maps, actions = maps.to(device), actions.to(device)
                outputs = model(maps)
                loss = criterion(outputs, actions)
                val_loss += loss.item() * actions.size(0)
                _, predicted = outputs.max(1)
                val_total += actions.size(0)
                val_correct += predicted.eq(actions).sum().item()
        avg_val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # Print only per epoch (not per batch)
        print(f"Epoch {epoch+1}/{num_epochs} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f} | Train acc: {train_acc:.2f}% | Val acc: {val_acc:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model_state = model.state_dict().copy()
            print(f'New best model with {best_accuracy:.2f}% val accuracy!')

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f'Final model: best val accuracy {best_accuracy:.2f}%')

    # Print summary for analysis
    print("\n=== Training summary ===")
    print("Epoch | Train loss | Val loss | Train acc | Val acc")
    for i in range(num_epochs):
        print(f"{i+1:5d} | {train_losses[i]:10.4f} | {val_losses[i]:8.4f} | {train_accuracies[i]:9.2f}% | {val_accuracies[i]:7.2f}%")
    print("=======================")
    print("\nInterpretation tips:")
    print("Train loss ↓, Val loss ↑ or steady → overfitting")
    print("Train and Val loss ↓, but slowly → still learning, but slow")
    print("Val metric (acc) grows a bit, loss drops → normal, training almost done")
    return model

def save_model(model, input_size, model_path="models/pacman_model.pth"):
    """Guarda el modelo entrenado"""
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    
    # Guardar el modelo junto con información sobre el tamaño de entrada
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
    }
    torch.save(model_info, model_path)
    print(f'Modelo guardado en {model_path}')

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for maps, actions in test_loader:
            maps, actions = maps.to(device), actions.to(device)
            outputs = model(maps)
            loss = criterion(outputs, actions)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += actions.size(0)
            test_correct += predicted.eq(actions).sum().item()
    test_accuracy = 100. * test_correct / test_total
    avg_loss = test_loss / len(test_loader)
    print(f"Test accuracy: {test_accuracy:.2f}% | Test loss: {avg_loss:.4f}")

def main():
    import time
    start_time = time.time()
    # Verificar disponibilidad de GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar datos
    maps, actions = load_and_merge_data()
    
    # Preprocesar mapas
    maps, input_size = preprocess_maps(maps)
    
    
    # Dividir en conjunto de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        maps, actions, test_size=0.2, random_state=PACMAN_SEED, stratify=actions
    )
    
    # Crear datasets
    train_dataset = PacmanDataset(X_train, y_train)
    test_dataset = PacmanDataset(X_test, y_test)
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Crear modelo
    model = PacmanNet(input_size, HIDDEN_SIZE, NUM_ACTIONS).to(device)
    print(f"Modelo creado: {model}")
    
    # Entrenar modelo
    trained_model = train_model(model, train_loader, test_loader, device)
    
    # Guardar modelo
    save_model(trained_model, input_size)
    # Проверка модели на тесте
    evaluate_model(trained_model, test_loader, device)
    print(f"Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")
if __name__ == "__main__":
    main()