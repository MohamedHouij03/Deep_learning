# ⚽ Football Image Understanding: From Bounding-Box Crops to End-to-End Object Detection

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Framework-005088?style=for-the-badge&logo=cpu" alt="Deep Learning Framework">
  <img src="https://img.shields.io/badge/YOLOv8-Success-11caa0?style=for-the-badge&logo=ultralytics" alt="YOLOv8 Success">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Gradio-UI%20Deployed-orange?style=for-the-badge&logo=gradio" alt="Gradio Deployed">
</p>

---

## 📌 1. Description du Problème

L'analyse automatique de flux vidéo de football se heurte à plusieurs défis majeurs : la forte densité de joueurs dans des zones restreintes, les occlusions partielles, les variations de luminosité et la vitesse élevée du ballon. 

Pour résoudre ce problème de manière rigoureuse, notre approche a été divisée en deux phases distinctes :
* 🔬 **Phase Académique Expérimentale :** Une approche modulaire consistant à isoler des vignettes (*crops*) de dimensions $64 \times 64$ pixels afin d'évaluer la capacité intrinsèque d'architectures standards (**MLP, CNN, RNN, LSTM**) à classifier les types d'entités de manière isolée.
* 🚀 **Phase de Détection Globale (End-to-End) :** Le déploiement d'une architecture unifiée de pointe (**YOLOv8**) capable de réaliser simultanément la localisation spatiale et la classification des entités directement sur l'image entière en temps réel.

---

## 📊 2. Pipeline et Structure du Jeu de Données

Les données proviennent de la plateforme **Roboflow** (Projet : `testing_football`, Workspace : `mohamed-houij`).

### 📦 Volume et Répartition des Données
* 🟩 **Entraînement (*Train Set*) :** 5 934 images uniques contenant 12 986 annotations de boîtes englobantes.
* 🟨 **Validation (*Val Set*) :** 602 images contenant 1 478 annotations.
* 🟥 **Test (*Test Set*) :** 149 images contenant 371 annotations.

### 🏷️ Classes Cibles (11 au total)
Le dataset natif regroupe 11 labels qui ont été cartographiés et standardisés par notre pipeline :
`Arbitre`, `Ballon`, `Football`, `Gardien`, `Joueur`, `Player`, `ball`, `big`, `person`, `player`, `refere`.

### ⚙️ Prétraitement : Fonction `safe_crop`
Pour la phase de classification, un script personnalisé extrait les coordonnées de chaque boîte englobante, applique des marges de sécurité pour éviter les sorties de matrice, normalise les intensités de pixels dans la plage $[0.0, 1.0]$ en `float32`, et redimensionne l'image au format standard **64x64x3**.

---

## 🔬 3. Architectures Expérimentées & Méthodologie

Quatre types d'architectures ont fait l'objet d'un étalonnage rigoureux sur la tâche de classification de crops :
1. **Multi-Layer Perceptrons (MLP) :** Établissement d'une baseline dense et plate au niveau des pixels.
2. **Ajustement des Hyperparamètres :** Évaluation comparative des fonctions d'activation (ReLU et variantes), optimisation des taux d'apprentissage, et intégration de la régularisation (Dropout et pénalité $L_2$).
3. **Convolutional Neural Networks (CNN) :** Extraction de caractéristiques spatiales pour capturer la géométrie locale et les motifs texturaux des maillots.
4. **Réseaux Récurrents (Simple RNN & LSTM) :** Traitement des lignes ou patchs de l'image sous forme de séquences pseudo-temporelles afin d'évaluer la mémoire de contexte spatial.

---

## 🏆 4. Performances et Résultats Établis

### 🎯 Classification de Crops (Phase 1)

| Icône | Modèle / Architecture | Précision Globale (*Accuracy*) | Type de Features Exploité |
| :---: | :--- | :---: | :--- |
| 👑 | **LSTM Baseline** | **77.1%** | Séquence de pixels (lignes) |
| 🖼️ | **CNN Baseline** | **70.6%** | Spatio-temporel (Convolutions 2D) |
| 🧠 | **MLP Dense** | **63.1%** | Vecteur de pixels aplati |
| 🔄 | **Simple RNN** | **48.2%** | Séquence basique sans porte |

### ⚡ Détection End-to-End YOLOv8 (Phase 2)

Suite aux expérimentations sur les vignettes, le modèle **YOLOv8 Nano (YOLOv8n)** a été entraîné sur les images complètes pendant **50 époques**. Malgré les ajustements de chemins d'accès locaux requis lors du déploiement final, les métriques d'évaluation valident l'efficacité du modèle :

* **Validation mAP50 :** `42.79%`
* **Test mAP50 :** `66.18%`
* **Vitesse d'inférence :** `~8.2 ms` par image sur GPU Tesla T4.

---

## 💻 5. Guide d'Exécution du Code

### 🛠️ Installation des Dépendances
Assurez-vous de disposer de Python 3.10+ et d'un environnement équipé d'un GPU (recommandé pour YOLOv8).

```bash
pip install ultralytics roboflow gradio tensorflow torch matplotlib numpy
