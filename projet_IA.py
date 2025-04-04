"""
Fiche de code principale poru le projet IA
"""
if __name__ == '__main__': 
    from multiprocessing import freeze_support                                  # Pour éviter les erreurs de multiprocessing sous Windows
    freeze_support() 

# Section des importations

    import os                                                                   # Pour les opérations système
    import torch                                                                # Pour le calcul numérique 
    import warnings                                                             # Pour les avertissements
    import rasterio                                                             # Pour les images
    import numpy as np                                                          # Pour les calculs numériques	
    import torchvision                                                          # Pour les modèles
    import itertools                                                            # Pour les itérations
    from math import *                                                          # Pour les fonctions mathématiques
    import pandas as pd                                                         # Pour les dataframes
    import seaborn as sns                                                       # Pour les graphiques
    from tqdm import tqdm                                                       # Pour les barres de progression
    import torch.nn as nn                                                       # Pour les réseaux de neurones   
    import torch.optim as optim                                                 # Pour les optimiseurs                            
    import matplotlib.pyplot as plt                                             # Pour les graphiques
    from eoreader.reader import Reader                                          # Pour lire les images
    from skimage import filters, color                                          # Pour les filtres
    from PIL import ImageEnhance, Image                                         # Pour les images
    from torchvision.models import resnet18                                     # Pour les modèles
    from torch.utils.data import DataLoader, random_split                       # Pour charger les données
    from torchvision.utils import make_grid                                     # Pour afficher les images
    import torchvision.transforms as transforms                                 # Pour les transformations
    from torchvision import datasets, transforms, models                        # Pour les modèles
    from torchvision.transforms.functional import to_pil_image                  # Pour convertir un tensor en image
    from sklearn.metrics import confusion_matrix, accuracy_score                # Pour les métriques    
    from eoreader.bands import RED, GREEN, NDVI, YELLOW, CLOUDS, to_str         # Pour les bandes

    #Function to choose the device
    def get_device():
        if torch.cuda.is_available():
            #Use CUDA device if available
            return torch.device("cuda"), "GPU"
        elif torch.backends.mps.is_available():
            #Use Apple Metal if available on Mac
            return torch.device("cuda"), "Apple Metal"
        else:
            # default to CPU
            return torch.device("cuda"), "CPU"

    #Setting the device
    device, device_name = get_device()
    print(f"Using device: {device_name}")


    """
    ==========================================================================================================
                            PARTIE 1 - PRETRAITEMENT & AFFICHAGE DES CLASSES ET SOUS-CLASSES
    ==========================================================================================================
    """
    
        #Transformation sequence to resize, crop, convert to tensor, and normalize the images
    transform = transforms.Compose([
        transforms.Resize(256),  #Resize each image to 256x256 pixels
        transforms.CenterCrop(224),  #Crop the central 224x224 area
        transforms.ToTensor(),  #Convert image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #Normalize tensor data
    ])

    #chargement des données

    #Load images from structured directory with automatic labeling based on folder names and apply transformations
    data_directory = '.\\RSI-CB128'
    dataset = datasets.ImageFolder(root=data_directory, transform=transform)


    # Initialisation du dictionnaire pour stocker les classes et sous-classes
    class_hierarchy = {}

    # Parcours des fichiers du dataset
    for path, index in dataset.samples:  # dataset.samples contient (chemin, index de classe)
        # Extraire le chemin relatif par rapport au dossier racine
        relative_path = os.path.relpath(path, data_directory)

        # Découper les dossiers pour identifier classe principale et sous-classe
        parts = relative_path.split(os.sep)  # Séparation selon "/"
        if len(parts) < 2:
            continue  # Ignorer les fichiers mal placés

        class_principale, sous_classe = parts[0], parts[1]  # Ex: ('animaux', 'chat')

        # Initialiser la classe principale si elle n'existe pas
        if class_principale not in class_hierarchy:
            class_hierarchy[class_principale] = {}

        # Ajouter la sous-classe et compter les images
        if sous_classe in class_hierarchy[class_principale]:
            class_hierarchy[class_principale][sous_classe] += 1
        else:
            class_hierarchy[class_principale][sous_classe] = 1

    # Affichage de l'organisation des classes et sous-classes
    print("\nOrganisation des classes et sous-classes :")
    for class_principale, sous_classes in class_hierarchy.items():
        print(f"\nClasse principale : {class_principale}")
        for sous_classe, count in sous_classes.items():
            print(f"  - {sous_classe}: {count} images")

    # Calcul total d'images
    total_images = sum(sum(sous_classes.values()) for sous_classes in class_hierarchy.values())
    print(f"\nTotal d'images dans le dataset : {total_images}")


    """
    ==========================================================================================================
                                    "     PARTIE 2 - ENTRAINEMENT DU MODÈLE
    ==========================================================================================================
    """

    #DataLoader manages data shuffling and batching automatically
    batch_size = 32  #Defines how many samples are processed before the model updates
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    images, _ = next(iter(data_loader))

    #Display the number of images
    print("------ IMAGES ------")
    print(f'Total images: {len(dataset)}')
    print(f'Size of image: {images.shape}')

    def show_six_images(data_loader, dataset):
        # Fetch the first batch of images and labels
        images, classes = next(iter(data_loader))

        # Define the number of images to display
        num_images = 6

        # Create a figure with subplots in a 2x3 configuration
        fig, axes = plt.subplots(2, 3, figsize=(10, 10))  # 2 rows, 3 columns
        axes = axes.flatten()  # Flatten the 2D array of axes into 1D for easier iteration

        # Loop through the first six images (or less if the batch is smaller)
        for idx, ax in enumerate(axes):
            if idx < num_images:
                # Convert the tensor image to a NumPy array for display
                img = images[idx].permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)

                # Display the image
                ax.imshow(img)
                ax.set_title(f'Class: {dataset.classes[classes[idx]]}')  # Set the title to the class label
                ax.axis('off')  # Hide the axes
            else:
                ax.axis('off')  # Hide unused subplots

        plt.tight_layout()
        plt.show()

    show_six_images(data_loader,dataset)














        #Transformations for the training data (with augmentation)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  #Convert 1 channel grayscale (MNIST) to 3 channel for model ResNet-18 input
        transforms.RandomRotation(10, fill=(0,0,0)),  #Rotate by up to 10 degrees filling with 0
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  #Random translation
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])

    #Transformations for the testing data (without augmentation)
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  #Convert 1 channel grayscale (MNIST) to 3 channel for model ResNet-18 input
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])


   #Loading dataset
    train_dataset = datasets.ImageFolder(root=data_directory, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=data_directory, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    #Load a pretrained ResNet-18 model
    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    print(f'Numbers of features in last layer of the pretrained model :{num_features}')
    model.fc = nn.Linear(num_features, 10)  #Adjust for 10 output classes
    model = model.to(device)  #Move model to the appropriate device














    #Define the loss function and the optimizer

    criterion = nn.CrossEntropyLoss()  #Suitable for classification tasks
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  #Stochastic Gradient Descent





    def train_and_validate(model, criterion, optimizer, train_loader, test_loader, epochs=10):
        train_losses = []  #List to store average training losses per epoch
        val_losses = []  #List to store average validation losses per epoch

        #Loop over the dataset for a fixed number of epochs
        for epoch in range(epochs):
            model.train()  #Set the model to training mode (enables dropout, batchnorm updates)
            total_train_loss = 0

            for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", leave=True):
                data, targets = data.to(device), targets.to(device)  #Move data to the correct device
                optimizer.zero_grad()  #Clear gradients before calculating new ones
                outputs = model(data)  #Forward pass: compute predicted outputs
                loss = criterion(outputs, targets)  #Calculate loss between predicted and true labels
                loss.backward()  #Backward pass: compute gradient of the loss wrt model parameters
                optimizer.step()  #Perform a single optimization step (parameter update)

                #Accumulate the loss over each batch
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            #Validation phase
            model.eval()  #Set the model to evaluation mode (disables dropout, batchnorm doesn't update)
            total_val_loss = 0

            with torch.no_grad():  #Disabling gradient calculation for validation (saves memory and computations)
                for data, targets in tqdm(test_loader, desc="Validating", leave=True):
                    data, targets = data.to(device), targets.to(device)  #Move data to the correct device
                    outputs = model(data)  #Forward pass: compute predicted outputs
                    loss = criterion(outputs, targets)  #Calculate loss between predicted and true labels
                    
                    #Accumulate the loss over each batch
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(test_loader)
            val_losses.append(avg_val_loss)

            #Output the average losses for the current epoch
            print(f'Train Loss: {np.mean(train_losses):.4f}, Validation Loss: {np.mean(val_losses):.4f}')
            print('---------------------------------------------')

        #Save the model to disk after training is complete
        torch.save(model.state_dict(), 'model.pth')
        print('Model saved to model.pth')

        #Return lists of average losses for each epoch for both training and validation
        return train_losses, val_losses
    
    train_losses, val_losses = train_and_validate(model, criterion, optimizer, train_loader, test_loader, epochs=5)

    def plot_training_progress(train_losses, val_losses):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, 'ro-', label='Training Loss')
        plt.plot(epochs, val_losses, 'bo-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    plot_training_progress(train_losses, val_losses)

        #Evaluate the model
    def evaluate_model(model, test_loader):
        model.eval()  #Set the model to evaluation mode, which disables dropout and batch normalization
        model.to(device)
        val_loss = 0  #Initialize validation loss
        correct = 0  #Count of correct predictions
        total = 0  #Total number of examples processed
        all_preds = []  #List to store all predictions
        all_labels = []  #List to store all true labels

        with torch.no_grad():  #Disable gradient computation to speed up the process and reduce memory usage
            for images, labels in test_loader:  #Iterate over all batches in the test dataset
                images, labels = images.to(device), labels.to(device)  #Move images and labels to the specified device
                outputs = model(images)  #Compute the model's output
                loss = criterion(outputs, labels)  #Calculate the loss based on the criterion
                val_loss += loss.item()  #Accumulate the loss

                _, predicted = torch.max(outputs.data, 1)  #Get the predicted classes (highest output values)
                total += labels.size(0)  #Update total count of examples
                correct += (predicted == labels).sum().item()  #Update correct count based on how many predictions match the labels

                all_preds.extend(predicted.cpu().numpy())  #Store predictions in a list
                all_labels.extend(labels.cpu().numpy())  #Store true labels in a list

        avg_val_loss = val_loss / len(test_loader)  #Calculate average validation loss
        avg_accuracy = correct / total  #Calculate accuracy as the ratio of correct predictions to total examples

        return avg_val_loss, avg_accuracy, all_preds, all_labels  #Return the computed metrics and lists of predictions and labels


    #Import model
    model = resnet18(pretrained=False) 
    #Adapt it to the number of classes on the last layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10) 
    #Load the model saved
    state_dict = torch.load('model.pth')
    model.load_state_dict(state_dict)

    test_loss, test_accuracy, all_preds, all_labels = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%, Test Loss: {test_loss:.4f}")


    