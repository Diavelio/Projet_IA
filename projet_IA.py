"""
Fiche de code principale poru le projet IA
"""
if __name__ == '__main__': 
    from multiprocessing import freeze_support                                  # Pour éviter les erreurs de multiprocessing sous Windows
    freeze_support() 

# Section des importations

    import os                                                                   # Pour les opérations système
    import torch                                                                # Pour les réseaux de neurones
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
    from torch.utils.data import DataLoader                                     # Pour charger les données
    from torchvision.utils import make_grid                                     # Pour afficher les images
    import torchvision.transforms as transforms                                 # Pour les transformations
    from torchvision import datasets, transforms, models                        # Pour les modèles
    from torchvision.transforms.functional import to_pil_image                  # Pour convertir un tensor en image
    from sklearn.metrics import confusion_matrix, accuracy_score                # Pour les métriques    
    from eoreader.bands import RED, GREEN, NDVI, YELLOW, CLOUDS, to_str         # Pour les bandes

    """
    ==========================================================================================================
                                                     PARTIE 1
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


