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
    from math import *                                                          # Pour les fonctions mathématiques
    import pandas as pd                                                         # Pour les dataframes
    import matplotlib.pyplot as plt                                             # Pour les graphiques
    from eoreader.reader import Reader                                          # Pour lire les images
    from skimage import filters, color                                          # Pour les filtres
    from PIL import ImageEnhance, Image                                         # Pour les images
    from torch.utils.data import DataLoader                                     # Pour charger les données
    from torchvision.utils import make_grid                                     # Pour afficher les images
    from torchvision import datasets, transforms, models                        # Pour les modèles
    from torchvision.transforms.functional import to_pil_image                  # Pour convertir un tensor en image
    from eoreader.bands import RED, GREEN, NDVI, YELLOW, CLOUDS, to_str         # Pour les bandes


    a = 1+1
    print(a)