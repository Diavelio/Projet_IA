
"""
Important : il faut que le fichier code soit dans le même dossier que le dossier files pour que le code marche
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

    #--------------------Exercice 1------------------------





    warnings.filterwarnings('ignore', category=DeprecationWarning)  # Pour ne pas avoir les alertes de bibliothèques


    #Transformation sequence to resize, crop, convert to tensor, and normalize the images
    transform = transforms.Compose([
        transforms.Resize(256),  #Resize each image to 256x256 pixels
        transforms.CenterCrop(224),  #Crop the central 224x224 area
        transforms.ToTensor(),  #Convert image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #Normalize tensor data
    ])


    #Load images from structured directory with automatic labeling based on folder names and apply transformations
    data_directory = '.\\files\\files\\aircraft' #C:\\Users\\romai\\Desktop\\projetIA
    dataset = datasets.ImageFolder(root=data_directory, transform=transform)


    #Display class indices and names
    print("Class indices:", dataset.class_to_idx)

    #Count the number of images per class
    class_counts = {}
    for _, index in dataset.samples:  #dataset.samples contains paths and class indices
        class_name = dataset.classes[index]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1

    #Display the number of files per class
    print("Number of images per class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    #Calculate total number of images
    total_images = sum(class_counts.values())
    print(f"Total number of images: {total_images}")



    #DataLoader manages data shuffling and batching automatically
    batch_size = 32  #Defines how many samples are processed before the model updates
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)



    def display_comparison(original, transformed, title):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        ax1.imshow(original)
        ax1.set_title('Original')
        ax1.axis('off')
        ax2.imshow(transformed, cmap = 'gray')
        ax2.set_title(title)
        ax2.axis('off')
        plt.show()


    # Fetch the first batch of images and labels
    images, classes = next(iter(data_loader))

    #Define the number of images to display
    num_images = 3

    #Create a figure with subplots in a 2x3 configuration
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))  #1 row, 3 columns
    axes = axes.flatten()  #Flatten the 2D array of axes into 1D for easier iteration

    #Loop through the first six images (or less if the batch is smaller)
    for idx, ax in enumerate(axes):
        if idx < num_images:
            #Convert the tensor image to a NumPy array for display
            img = images[idx].numpy().transpose((1, 2, 0))  #Change from (C, H, W) to (H, W, C)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean  #Undo the normalization
            img = np.clip(img, 0, 1)  #Ensure values are within [0, 1] to display correctly

            # Display the image
            """
            ax.imshow(img)
            ax.set_title(f'Class: {dataset.classes[classes[idx]]}')  #Set the title to the class label
            ax.axis('off')  #Hide the axes
            """



            # Load the original image

            original_img = Image.fromarray((img * 255).astype(np.uint8))  #Image.open('/content/drive/MyDrive/files/aircraft/real_private_aircraft/pri14.jpg')
            resized_img = original_img.resize((128,128))  #Resize to 128x128 pixels

            display_comparison(img, resized_img, 'Resized Image')

                    # Convert the image to grayscale
            grayscale_img = original_img.convert('L')  # 'L' mode represents grayscale

            resized_img = grayscale_img.resize((128, 128))  # Resize to 128x128 pixels

            display_comparison(original_img, resized_img, 'Resized Image (Grayscale)')

        else:
            ax.axis('off')  # ide unused subplots




    #plt.tight_layout()
    #plt.show()


    """
    # Load the original image
    original_img = Image.fromarray((img[0] * 255).astype(np.uint8)) # Convert to PIL Image
    resized_img = original_img.resize((128, 128))  #Resize to 128x128 pixels
    """


    #--------------------Exercice 3------------------------

    def display_comparison(original, transformed, title):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(original)
        ax1.set_title('Original')
        ax1.axis('off')
        ax2.imshow(transformed)
        ax2.set_title(title)
        ax2.axis('off')
        plt.show()
    
    def is_valid_file(filename):
        return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.JPG', '.JPEG'))



    # Load the original image
    original_img = Image.open('.\\files\\files\\aircraft\\real_private_aircraft\\pri14.jpg')
    resized_img = original_img.resize((128, 128))  #Resize to 128x128 pixels

    display_comparison(original_img, resized_img, 'Resized Image')

    from PIL import ImageEnhance

    original_img = Image.open('.\\files\\files\\aircraft\\real_private_aircraft\\pri15.jpg')

    #Brightness enhancement
    enhancer = ImageEnhance.Brightness(original_img)
    bright_img = enhancer.enhance(2.0)  # Increase brightness

    #Contrast enhancement
    enhancer = ImageEnhance.Contrast(original_img)
    contrast_img = enhancer.enhance(2.0)  # Increase contrast

    display_comparison(original_img, bright_img, 'Brightened Image')
    display_comparison(original_img, contrast_img, 'Enhanced Contrast Image')

    # Transformation sequence to resize, crop, convert to tensor, and normalize the images
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize each image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the central 224x224 area
        transforms.ToTensor(),  # Convert image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize tensor data
    ])

    #Load images from structured directory with automatic labeling based on folder names and apply transformations
    data_directory = '.\\files\\files\\aircraft'
    dataset = datasets.ImageFolder(root=data_directory, transform=transform)

    def flip_horizontal(image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def flip_vertical(image):
        return image.transpose(Image.FLIP_TOP_BOTTOM)

    # Charger l'image originale
    original_img = Image.open('.\\files\\files\\aircraft\\real_private_aircraft\\pri16.jpg')

    # Retourner l'image horizontalement
    flipped_horizontal_img = flip_horizontal(original_img)
    display_comparison(original_img, flipped_horizontal_img, 'Retournement Horizontal Image')

    # Retourner l'image verticalement
    flipped_vertical_img = flip_vertical(original_img)
    display_comparison(original_img, flipped_vertical_img, 'Retournement Vertical Image')

    def compare_all_images(directory, transform):
        for root, _, files in os.walk(directory):
            for file in files:
                if is_valid_file(file):
                    file_path = os.path.join(root, file)
                    original_img = Image.open(file_path)
                    
                    # Appliquer les transformations
                    resized_img = original_img.resize((128, 128))
                    bright_img = ImageEnhance.Brightness(original_img).enhance(2.0)
                    contrast_img = ImageEnhance.Contrast(original_img).enhance(2.0)
                    flipped_horizontal_img = flip_horizontal(original_img)
                    flipped_vertical_img = flip_vertical(original_img)
                    
                    # Afficher les comparaisons
                    display_comparison(original_img, resized_img, 'Resized Image')
                    display_comparison(original_img, bright_img, 'Brightened Image')
                    display_comparison(original_img, contrast_img, 'Enhanced Contrast Image')
                    display_comparison(original_img, flipped_horizontal_img, 'Flipped Horizontal Image')
                    display_comparison(original_img, flipped_vertical_img, 'Flipped Vertical Image')

    # Utilisation de la fonction
    #data_directory = '.\\files\\files\\aircraft'
    #compare_all_images(data_directory, transform)