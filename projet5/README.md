
# Segmentez des clients d'un site e-commerce

Vous êtes consultant pour  [Olist](https://olist.com/), une entreprise brésilienne qui propose une solution de vente sur les marketplaces en ligne.

[![La page d'accueil du site Olist](https://user.oc-static.com/upload/2019/02/24/15510251487267_Capture%20d%E2%80%99e%CC%81cran%202019-02-20%20a%CC%80%2017.37.38.png)](https://user.oc-static.com/upload/2019/02/24/15510251487267_Capture%20d%E2%80%99e%CC%81cran%202019-02-20%20a%CC%80%2017.37.38.png)

La page d'accueil du site Olist

**Olist**  souhaite que vous fournissiez à ses équipes d'e-commerce une  **segmentation des clients**  qu’elles pourront utiliser au quotidien pour leurs campagnes de communication.

Votre objectif est de  **comprendre les différents types d’utilisateurs**  grâce à leur comportement et à leurs données personnelles.

Vous devrez  **fournir à l’équipe marketing une description actionable**  de votre segmentation et de sa logique sous-jacente pour une utilisation optimale, ainsi qu’une  **proposition de contrat de maintenance**  basée sur une analyse de la stabilité des segments au cours du temps.

### **Les données**

Pour cette mission, Olist vous fournit une  [base de données](https://www.kaggle.com/olistbr/brazilian-ecommerce)  anonymisée comportant des informations sur l’historique de commandes, les produits achetés, les commentaires de satisfaction, et la localisation des clients depuis janvier 2017.

### **Votre mission**

Votre mission est d’aider les équipes d’Olist à comprendre les différents types d'utilisateurs. Vous utiliserez donc des méthodes non supervisées pour regrouper des clients de profils similaires. Ces catégories pourront être utilisées par l’équipe Marketing pour mieux communiquer.

Vous créez donc un notebook et démarrez votre travail d’analyse exploratoire.

**![](https://user.oc-static.com/upload/2021/12/01/16383463579756_DATA_unesemaineplustard_IT_FR_UneSemaine%20copy%20%282%29.png)**

Après quelques premières analyses, vous vous rendez compte qu’Olist ne vous a pas fourni beaucoup de données ; vous enquêtez donc auprès de l’entreprise pour obtenir quelques informations complémentaires, et vérifier que vous avez bien compris la mission. Voici sa réponse.

Bonjour,

Pour des raisons de confidentialité, nous ne pouvons pas vous fournir beaucoup de données à ce stade. Ensuite, en raison de ressources limitées, nous avons dû vous fournir l’ensemble des données, alors que seule une partie va vous intéresser. Nos dashboards internes nous indiquent en effet que seuls 3 % des clients du fichier de données partagé avec vous ont réalisé plusieurs commandes.

Nous sommes confiants sur le fait que les données à disposition suffiront pour réaliser un premier clustering. Cela a déjà été fait par d’autres prestataires par le passé, avec encore moins de données.

La segmentation proposée doit être exploitable et facile d’utilisation par notre équipe Marketing. Elle doit au minimum pouvoir différencier les bons et moins bons clients en termes de commandes et de satisfaction. Nous attendons bien sûr une segmentation sur l’ensemble des clients.

Dans un deuxième temps, une fois le modèle de segmentation choisi, nous souhaiterions que vous nous fassiez une recommandation de fréquence à laquelle la segmentation doit être mise à jour pour rester pertinente, afin de pouvoir effectuer un devis de contrat de maintenance.

Pour information, le code fourni doit respecter la convention PEP8, pour être utilisable par Olist.

Bien à vous,

Juan, de l’équipe Marketing

Vous réalisez donc trois autres documents afin de préparer une réponse à Olist :

-   un notebook avec des essais des différentes approches de modélisation ;
-   un notebook de simulation pour déterminer la fréquence nécessaire de mise à jour du modèle de segmentation, afin que celui-ci reste pertinent ;
-   une présentation pour un collègue afin d’obtenir ses retours sur votre approche.

### **Livrables**

-   Un  **notebook de l'analyse exploratoire**  (non cleané, pour comprendre votre démarche).
-   Un  **notebook**  (ou code commenté au choix)  **d’essais**  des différentes approches de modélisation (non cleané, pour comprendre votre démarche).
-   Un  **notebook de simulation pour déterminer la fréquence nécessaire de mise à jour**  du modèle de segmentation.
-   Un  **support de présentation**  pour présenter votre travail à un collègue.
