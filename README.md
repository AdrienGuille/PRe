# Projection 2D des mots vecteurs 

Cette application permet de visualiser en deux dimensions les mots représentés par des vecteurs de grande dimension générés par la bibliothèque FastText de gensim.

## Dépendances

Afin d'executer l'application, ces bibliothèques doivent être installées :

* [Bokeh](https://bokeh.pydata.org/) version >= 0.13
* [NumPy](http://www.numpy.org/) et [SciPy](https://www.scipy.org/)
* [Scikit-Learn](http://scikit-learn.org/)
* [Gensim](https://radimrehurek.com/gensim/index.html)
* [Node.js](https://nodejs.org/) version >= 6.10.0

Il est recommandé d'utiliser le logiciel [Anaconda](https://www.anaconda.com/) afin d'installer les dépendances mentionnés ci-dessus.

## Execution de l'application

Afin d'executer l'application et créer le serveur, il suffit d'utiliser la commande suivante :
```
bokeh serve nom_du_dossier
```
Remplacer 'nom_du_dossier' par le nom du dossier contentant les fichiers de l'application, notamment le fichier main.py.

N.B: Si vous voulez afficher la page web de l'application, il suffit d'aller vers 'localhost:5006' ou ajouter '--show' à la commande ci-dessus : 
```
bokeh serve --show nom_du_dossier
```
## Modéles

Cette application cherche automatiquement tout les modèles présents dans le dossier gensimModels et les ajoute dans la liste des modèles dans la page web.

Les modèles utilisés par l'application doivent être généré par la bibliothèque fasttext de gensim.
