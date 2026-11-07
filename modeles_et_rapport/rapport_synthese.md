# Rapport de synthèse : classification d'images Intel

Jeu de données : Intel Image Classification (6 classes). Entraînement : 11228 images, validation : 2806, test : 3000.
Protocole identique pour les trois modèles : mêmes données, EarlyStopping sur la val_loss (patience 8, poids restaurés), au plus 40 epochs.

## Résultats

| Modèle | Paramètres | Epochs | Temps (min) | Acc. validation | Loss validation | Acc. test | Loss test | F1 macro (test) |
|---|---|---|---|---|---|---|---|---|
| Modèle 1 (CNN initial) | 101,894 | 40 | 7.2 | 0.8806 | 0.322 | 0.881 | 0.3531 | 0.8824 |
| Modèle 2 (CNN amélioré) | 406,694 | 19 | 9.8 | 0.8175 | 0.4852 | 0.8053 | 0.5375 | 0.8063 |
| Modèle 3 (MobileNetV2) | 2,265,670 | 65 | 22.0 | 0.9205 | 0.2173 | 0.919 | 0.2266 | 0.9205 |

## Conclusions

- Modèle retenu d'après la validation : **Modèle 3 (MobileNetV2)** (val_loss = 0.2173, accuracy de validation = 0.9205).
- Accuracy de test du modèle retenu : **91.90 %**, soit +3.80 points par rapport au CNN initial (88.10 %).
- Temps d'entraînement : 22.0 min pour le modèle retenu contre 7.2 min pour le CNN initial.

## Limites

- Un seul entraînement par modèle : un écart d'un point environ entre deux exécutions n'est pas significatif.
- Marge d'erreur d'une accuracy mesurée sur 3000 images : environ ±1,3 point (intervalle à 95 %).
- L'accuracy d'entraînement affichée dans les courbes des modèles 2 et 3 est mesurée avec augmentation et Dropout actifs.
