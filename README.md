# dc-chimie

Prédiction de pIC50 à partir de molécules encodées en SMILES. Données fournies :
4 400 molécules d'entraînement et 2 934 molécules de test.

## Installation avec uv

Depuis la racine du projet, dans PowerShell :

```powershell
uv sync --locked
uv run python --version
uv run jupyter lab
```

`uv sync` crée automatiquement `.venv` et installe les dépendances de `uv.lock`,
y compris Jupyter. Python 3.14 est sélectionné par `.python-version`.
Il n'est pas nécessaire d'activer le venv pour utiliser `uv run`.
Dans VS Code, sélectionner `.venv\Scripts\python.exe` comme interpréteur et
comme noyau des notebooks.

Activation manuelle facultative :

```powershell
.\.venv\Scripts\Activate.ps1
```

## Organisation

```text
data/raw/                CSV fournis, noms et contenu conservés
data/processed/          Emplacement des futures données calculées
notebooks/
  reference/             Notebook fourni, conservé sans modification
  00_baseline.ipynb      Exécution de la baseline reproductible
  01_exploration.ipynb   Statistiques et contrôles des données
src/dc_chimie/
  data.py                Chargement et alignement des cibles par id
  features.py            Empreintes Morgan et contrôle des SMILES
  validation.py          Folds et métriques partagés
  train.py               Validation, entraînement et suivi des expériences
  predict.py             Soumission depuis un modèle sauvegardé
configs/baseline.yaml    Paramètres du modèle, des empreintes et de la validation
outputs/
  experiments.csv        Historique des expériences, créé au premier entraînement
  runs/                  Paramètres, scores par fold, versions et empreintes des CSV
  models/                Modèles entraînés
  submissions/           CSV de prédictions
```

Le notebook de référence garde ses anciens chemins et sorties pour archivage.
Utiliser `00_baseline.ipynb` pour exécuter la version adaptée au projet.
Les fichiers générés et `.venv` sont exclus de Git ; `experiments.csv`, les
configurations et `uv.lock` peuvent être versionnés.

## Lancer une expérience

```powershell
uv run dc-train
# Variante avec une autre configuration :
uv run dc-train --config configs/baseline.yaml
```

La baseline utilise des empreintes Morgan de rayon 2 sur 2 048 bits, sans
chiralité, et une Random Forest de 100 arbres. La seed vaut 42 ; la validation
utilise 5 folds mélangés. Les folds s'exécutent séquentiellement et chaque forêt
utilise les cœurs disponibles pour éviter deux niveaux de parallélisation.

Les scores rapportés sont les moyennes par fold de RMSE, MAE (erreur absolue
moyenne) et MedAE (erreur absolue médiane). Le notebook d'origine appelait
incorrectement la MedAE « MAE » et annonçait 10 folds au lieu de 5. Ses anciens
scores ne sont pas directement comparables à ceux du nouveau découpage.

Chaque exécution produit un identifiant unique partagé par le modèle, la
soumission et le fichier de métadonnées. La configuration complète est aussi
inscrite dans `outputs/experiments.csv`. Les données brutes restent inchangées.

Pour régénérer les prédictions d'une expérience (remplacer `<run_id>`) :

```powershell
uv run dc-predict outputs/models/<run_id>.joblib --output outputs/submissions/replay.csv
```

La reproductibilité suppose les mêmes données, la même configuration et le même
environnement verrouillé. Le découpage aléatoire constitue une baseline ; une
validation par familles structurales pourra compléter l'évaluation ensuite.

## Travailler sur de nouvelles idées

Explorer dans les notebooks, puis déplacer les traitements réutilisés dans
`src/dc_chimie/`. Copier la configuration pour comparer des variantes avec les
mêmes folds et métriques. Les CSV de `data/raw/` ne doivent pas être modifiés.

Documentation des empreintes : [API RDKit](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html).

## État de la vérification locale

L'environnement `.venv` et les dépendances ont été installés avec `uv sync`.
L'exécution complète de la baseline n'a pas pu être validée : Windows a bloqué
le chargement d'extensions natives NumPy/SciPy/RDKit avec le message
« Une stratégie de contrôle d'application a bloqué ce fichier ».
Aucun nouveau score ni modèle entraîné n'est donc annoncé. Ce blocage système
doit être résolu avant de pouvoir vérifier l'entraînement de bout en bout.
