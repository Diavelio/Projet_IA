Commandes git terminal : 

.



Cloner un répertoire : 

1 : créer le dossier avec le même chemin que les collaborateurs

2 : git clone lien -> cloner le répertoire référencé par le lien (il faut être collaborateur du projet pour y avoir accès)

.
.

Upload les modifications faites en local : 

1 : git add exemple.py         -> ajouter le fichier "exemple.py" à la file de commit

2 : git commit -m "message"    -> importe le fichier sur le hub avec le message entre guillemts (obligé d'en mettre un)

3 : git push                   -> écrit et rend disponible le fichier pour tous les collaborateurs du git
.
.


/!\ : ne pas push avant d'avoir au préalable pull, pour ne pas écraser d'éventuelles modifications faites par les autres collaborateurs

/!\ : ne pas faire "git add ." car tous les fichiers seront uploadés, et le site crashera.

.

git pull -> télécharger la version la plus récente du fichier ouvert 

.
.

Pour les chemins de fichiers, le ".\\fichier" indique qu'il prendra le fichier dans le même dossier que le code


git reset --hard HEAD~1       ->   annuler le dernier commit et effacer les modifs
.
git reset --soft HEAD~1       ->   annuler le dernier commit et garder les modifs

