# 🛡️ AdversarialLab — Plateforme de simulation d'attaques adversariales

Application web locale permettant de **tester la robustesse d'un modèle de classification** face aux
attaques adversariales, conformément au sujet « Robustesse des IA » :

| Phase du sujet | Implémentation |
|---|---|
| 1️⃣ Tester un modèle avec des données normales | Dataset synthétique de formes + CNN entraîné + évaluation (précision, matrice de confusion, F1) |
| 2️⃣ Simuler une attaque par **évasion** | **FGSM** (1 pas), **PGD** (itératif), bruit gaussien (baseline) + courbes précision/ε |
| 2️⃣ Simuler une attaque par **empoisonnement** | **Retournement d'étiquettes** + **porte dérobée (backdoor)** avec trigger visuel, ré-entraînement sur données corrompues |
| 3️⃣ Contrer l'attaque | **Entraînement adversarial** (Madry et al.), **prétraitements d'entrée** (quantification, flou, médian, JPEG), **détection/filtrage des données empoisonnées** + ré-entraînement |

Le tout dans une interface web à 6 onglets avec graphiques, images adversariales et métriques — en
français, pensée pour la démonstration et le rapport.

---

## ✨ Ce que montre l'application

- **Onglet 1 — Données & Modèle** : génération d'un dataset d'images synthétiques (cercle / carré /
  triangle / croix, ~40 Mo, entièrement aléatoire et local) puis entraînement d'un CNN avec courbe
  d'apprentissage.
- **Onglet 2 — Évaluation** : précision, matrice de confusion, précision/rappel/F1, exemples de
  prédictions. *C'est l'état « sain » de référence.*
- **Onglet 3 — Évasion** : FGSM / PGD / bruit avec force ε réglable. Visualisation original →
  adversarial → perturbation amplifiée, taux de succès de l'attaque et courbes précision vs ε.
- **Onglet 4 — Empoisonnement** : corruption du jeu d'entraînement (labels ou backdoor avec trigger),
  ré-entraînement, mesure du **taux de succès de la porte dérobée** — le modèle a une précision
  apparemment correcte mais trahit dès qu'on colle le motif.
- **Onglet 5 — Défenses** : 3 défenses évaluées chiffres à l'appui (gain sous attaque, coût sur
  données propres, précision/rappel de la détection d'empoisonnement avec histogramme des pertes).
- **Onglet 6 — Récapitulatif** : historique persistant de toutes les expériences — copier-coller
  direct dans le rapport.
- **Onglet 7 — Démo live** : chargement d'une **image depuis votre ordinateur** (glisser-déposer),
  classification en direct avec barres de confiance, puis attaque de cette image (PGD / FGSM /
  bruit / trigger backdoor) : la prédiction bascule sous vos yeux. Inclut l'**export des assets
  originaux** (images du dataset en PNG individuels, dans `data/assets_originaux/` ou en ZIP)
  et un scénario de démonstration guidé de ~3 minutes.

## 🚀 Démarrage rapide (WSL + GPU)

### 1. Prérequis
- Windows avec **WSL2** (`wsl --install`) et un driver NVIDIA récent (le GPU passe-through fonctionne
  nativement dans WSL2). *Sans GPU, tout fonctionne aussi sur CPU — simplement plus lent.*
- Python ≥ 3.10 dans WSL (`sudo apt install python3.10-venv` si besoin, ou installez [uv](https://docs.astral.sh/uv/) : `curl -LsSf https://astral.sh/uv/install.sh | sh`).

### 2. Lancer (une seule commande)
```bash
# dans WSL — depuis le dossier du projet
./run.sh
```
Le script crée le venv au premier lancement (`~/venvs/adversarial-lab`), installe les dépendances
(torch CUDA ~3 Go de téléchargement la première fois), puis démarre le serveur.

### 3. Ouvrir l'interface depuis Windows
➡️ **[http://localhost:8000](http://localhost:8000)** (le port est automatiquement transféré de WSL
vers Windows). La documentation interactive de l'API est sur
[http://localhost:8000/docs](http://localhost:8000/docs).

> ⚠️ Astuce performance : si le projet se trouve sur un disque Windows (`/mnt/...`), tout
> fonctionnera, mais un clone dans le home WSL (`~/`) sera plus rapide. Ici le dataset est chargé
> entièrement en mémoire, l'impact reste limité.

## 🧪 Vérifier le pipeline sans le serveur
```bash
~/venvs/adversarial-lab/bin/python scripts/smoke_test.py
```
Exécute toutes les briques (génération → entraînement → attaques → défenses) et affiche PASS/FAIL
par étape (~2 à 5 min selon le matériel).

## 👨‍👩‍👧‍👦 Tester chez soi (autres membres de l'équipe)

```bash
git clone https://github.com/<votre-org>/adversarial-lab.git
cd adversarial-lab

# Option A — script tout-en-un (Linux/WSL/macOS)
./run.sh

# Option B — venv classique
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Puis ouvrir http://localhost:8000. **Aucune donnée à télécharger** : le dataset est régénéré localement
par un clic dans l'interface (bouton « Générer le dataset », ~10 s). Sous Windows nativement
(sans WSL), la même séquence fonctionne — le GPU est détecté si CUDA est installé, sinon CPU.

## 🖥️ Ouvrir dans PyCharm

1. **File → Open** → sélectionner le dossier `adversarial-lab`.
2. **Settings → Project → Python Interpreter → Add Interpreter** :
   - *WSL* : choisir « WSL » → `/home/<user>/venvs/adversarial-lab/bin/python` (PyCharm Professional) ;
   - sinon pointer vers n'importe quel venv où `requirements.txt` est installé.
3. Une configuration d'exécution **« AdversarialLab (uvicorn) »** est déjà fournie dans `.run/` :
   bouton ▶️ vert → l'interface est sur http://localhost:8000.

## 🧠 Détails scientifiques

### Attaques d'évasion (inférence modifiée, modèle intact)
| Attaque | Principe | Paramètre |
|---|---|---|
| **FGSM** (Goodfellow et al. 2015) | un seul pas dans le sens du gradient signé : `x' = x + ε·sign(∇ₓ L)` | ε |
| **PGD** (Madry et al. 2018) | FGSM itératif avec pas plus petits, projection dans la boule L∞ et départ aléatoire — la référence | ε, itérations, α |
| Bruit gaussien | perturbation aléatoire **non optimisée** — baseline qui prouve que FGSM/PGD sont bien *adaptive* | ε |

### Attaques d'empoisonnement (données d'entraînement modifiées, modèle compromis)
- **Retournement d'étiquettes** : q % des labels sont remplacés (ciblé vers une classe, ou aléatoire).
- **Backdoor** : q % des images reçoivent un damier 6×6 en haut à gauche et sont relabelées vers la
  cible. Le modèle apprend *trigger ⇒ cible* ; on mesure ensuite le **taux de succès backdoor** :
  la fraction d'images de test portant le trigger classées dans la cible.

### Défenses
| Défense | Contre | Principe | Mesure |
|---|---|---|---|
| **Entraînement adversarial** | évasion | chaque batch est augmenté de ses exemples adversariaux (FGSM ou PGD) | précision propre vs sous attaque, avant/après |
| **Prétraitements** (quantification 4 niveaux, flou gaussien 5×5, filtre médian 3×3, JPEG q30) | évasion | détruisent la perturbation de haute fréquence sans ré-entraîner | gain sous attaque + coût sur données propres |
| **Détection par perte + filtrage** | empoisonnement | les exemples empoisonnés ont une perte atypique pour le modèle compromis (outliers statistiques, seuil z) → suppression → ré-entraînement | précision/rappel de détection, succès backdoor avant/après |

### Reproductibilité
Graine fixée (42 par défaut, modifiable) pour dataset, initialisation du modèle et ordre des
batches. Les artefacts (dataset NPZ, poids .pt, historique JSON) sont persistés dans `data/`
(ignoré par git) — redémarrer le serveur ne perd rien. NB : sur GPU, les opérations parallèles
introduisent une légère non-déterminisme d'exécution ; deux exécutions identiques donnent des
résultats très proches mais non bit-à-bit.

## 🗂️ Structure du projet
```
adversarial-lab/
├── app/
│   ├── main.py            # API FastAPI + orchestration des jobs d'arrière-plan
│   ├── state.py           # état global + persistance (dataset, modèles, historique)
│   └── ml/
│       ├── dataset.py     # génération du dataset synthétique de formes
│       ├── model.py       # CNN (batch norm, pooling adaptatif)
│       ├── engine.py      # entraînement/évaluation (device GPU/CPU auto)
│       ├── attacks.py     # FGSM, PGD, bruit gaussien
│       ├── poison.py      # label flipping, backdoor, succès backdoor
│       └── defense.py     # adv training, prétraitements, détection/filtrage
├── static/                # interface web (HTML/CSS/JS vanilla + Chart.js vendu, 7 onglets)
├── scripts/smoke_test.py  # test de bout en bout sans serveur
├── run.sh                 # démarrage 1 commande (WSL/Linux/macOS)
├── .run/                  # configuration d'exécution PyCharm
└── requirements.txt
```

## 📋 Prérequis matériels & limites connues
- RAM : ~2 Go libres suffisent ; disque : ~7 Go pour torch+CUDA (W)SL.
- GTX 1650 4 Go testée ✔ — tout GPU CUDA moderne fonctionne.
- Le filtrage d'empoisonnement se base sur la perte : efficace sur le retournement d'étiquettes,
  plus mitigé sur les backdoors « léchées » (limite documentée dans la littérature) — l'histogramme
  des pertes affiché permet de juger sur pièce.
- L'application est destinée à un usage local pédagogique (pas d'authentification, jobs séquentiels).
