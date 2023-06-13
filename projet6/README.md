# Enoncé

Vous êtes Data Scientist au sein de l’entreprise "**Place de marché**”, qui souhaite lancer une marketplace e-commerce.

[![logo entreprise place de marché](https://user.oc-static.com/upload/2019/02/24/15510259240381_Projet%20textimage%20logo.png)](https://user.oc-static.com/upload/2019/02/24/15510259240381_Projet%20textimage%20logo.png)

Sur la place de marché, des vendeurs proposent des articles à des acheteurs en postant une photo et une description.

Pour l'instant, l'attribution de la catégorie d'un article est effectuée manuellement par les vendeurs, et est donc peu fiable. De plus, le volume des articles est pour l’instant très petit.

Pour rendre l’expérience utilisateur des vendeurs (faciliter la mise en ligne de nouveaux articles) et des acheteurs (faciliter la recherche de produits) la plus fluide possible, et dans l'optique d'un passage à l'échelle, **il devient nécessaire d'automatiser cette tâche**.

**Linda**, Lead Data Scientist, vous demande donc d'étudier la faisabilité d'un  **moteur de classification**  des articles en différentes catégories, avec un niveau de précision suffisant.

Voici le mail qu’elle vous a envoyé.

Bonjour,

Merci pour ton aide sur ce projet !

Ta mission est de réaliser,  **dans une première itération, une étude de faisabilité d'un moteur de classification**  d'articles, basé sur une image et une description, pour l'automatisation de l'attribution de la catégorie de l'article.

Tu dois **analyser les descriptions textuelles et les images des produits**, au travers des étapes suivantes :

-   **Un prétraitement**  des données texte ou image suivant le cas ;
-   Une extraction de features ;
-   Une  **réduction en 2 dimensions**, afin de projeter les produits sur un graphique 2D, sous la forme de points dont la couleur correspondra à la catégorie réelle ;
-   **Analyse du graphique**  afin d’en déduire ou pas, à l’aide des descriptions ou des images, la  **faisabilité de regrouper automatiquement**  des produits de même catégorie ;
-   Réalisation d’une  **mesure pour confirmer ton analyse visuelle**, en calculant la similarité entre les catégories réelles et les catégories issues d’une segmentation en clusters.

Pourrais-tu nous démontrer, par cette approche, la faisabilité de regrouper automatiquement des produits de même catégorie ?

Voici les contraintes :

-   Afin d’extraire les features texte, il sera nécessaire de mettre en œuvre :

-   deux approches de type “bag-of-words”, comptage simple de mots et Tf-idf ;
-   une approche de type word/sentence embedding classique avec Word2Vec (ou Glove ou FastText) ;
-   une approche de type word/sentence embedding avec BERT ;
-   une approche de type word/sentence embedding avec USE (Universal Sentence Encoder).

En pièce jointe, tu trouveras un  **exemple de mise en œuvre de ces approches d’extraction de features texte sur un autre dataset**. Je t’invite à l’utiliser comme point de départ, cela va te faire gagner beaucoup de temps !

Afin d’extraire les features image, il sera nécessaire de mettre en œuvre :

-   un algorithme de type SIFT / ORB / SURF ;
-   un algorithme de type CNN Transfer Learning.

Concernant l’approche de type SIFT, je t’invite à regarder le webinaire que nous avons réalisé, disponible dans les ressources.

En pièces jointes, tu trouveras un exemple de mise en œuvre de l’approche de type CNN Transfer Learning d’extraction de features images sur un autre dataset. Je t’invite à l’utiliser comme point de départ, cela va te faire gagner beaucoup de temps !

Merci encore,

Linda

PS : J’ai bien vérifié qu’il n’y avait aucune contrainte de propriété intellectuelle sur les données et les images.

Pièces jointes :

-   [premier jeu de données d’articles avec le lien pour télécharger la photo et une description associée](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Textimage+DAS+V2/Dataset+projet+pre%CC%81traitement+textes+images.zip)
-   [un notebook d’exemple d’étude de faisabilité](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P6/Weather_Images_CNN_Transfer_Learning_Stage_1_feasibility_V1.0.ipynb)

![](https://user.oc-static.com/upload/2023/03/01/16776627204206_DATA_une-semaine-plus-tard.png)

Une semaine plus tard, vous partagez votre travail avec Linda, et elle vous répond avec enthousiasme avec une autre demande.

Bonjour,

Merci beaucoup pour ton travail !

Bravo d’avoir démontré la faisabilité de regrouper automatiquement des produits de même catégorie !

Maintenant, je te propose de passer à la deuxième itération. Pourrais-tu réaliser une  **classification supervisée à partir des images**  ? Je souhaiterais que tu mettes en place une  **data augmentation**  afin d’optimiser le modèle.

En pièce jointe, tu trouveras un exemple de mise en œuvre de classification supervisée sur un autre dataset. Je t’invite à l’utiliser comme point de départ, cela va te faire gagner beaucoup de temps !

Nous souhaitons élargir notre gamme de produits, en particulier dans l’épicerie fine. Pourrais-tu tester la collecte de produits à base de “champagne” via **l’**[**API disponible ici**](https://rapidapi.com/edamam/api/edamam-food-and-grocery-database) ? Je souhaiterais que tu puisses nous fournir une extraction des 10 premiers produits dans un fichier “.csv”, contenant pour chaque produit les données suivantes : foodId, label, category, foodContentsLabel, image.

Merci encore,

Linda

Pièces jointes :

-   [un notebook d’exemple de classification supervisée d’images](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P6/Weather_Images_CNN_Transfer_Learning_Stage_2_supervised_classification_V1.0.ipynb)
