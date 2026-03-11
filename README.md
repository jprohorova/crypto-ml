# crypto-ml
Dans une première expérience multi-classe, le modèle atteint une accuracy d’environ 40 %, supérieure au hasard (20 %), ce qui montre qu’il peut distinguer certaines familles de fonctions de hachage. Cependant, ce résultat est partiellement expliqué par des différences structurelles simples, notamment la longueur des sorties.
En revanche, lorsque la tâche est restreinte à SHA-256 et SHA3-256, l’accuracy reste proche du hasard (environ 52 %), ce qui suggère que les deux fonctions cryptographiques produisent des sorties difficilement séparables par un modèle simple.
