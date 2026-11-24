# ShopPredict

Application web qui estime la probabilité qu'une session de navigation sur un site e-commerce se termine par un achat.

Projet de fin d'année du cours Machine Learning in Production (EFREI).

## Équipe

- Candice Mamou : données et modèle
- Hugo Bouffier : application et Docker
- Côme Massaini : CI/CD, déploiement et réentraînement

## Données

[Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI, licence CC BY 4.0).

## Branches

```
feature/* → dev → staging → main
```

Chaque changement part d'une branche `feature/` créée depuis `dev` et passe par une pull request relue par un autre membre. Les branches `dev`, `staging` et `main` sont protégées.

## Application en ligne

https://shoppredict-o1z4.onrender.com

- `GET /` : formulaire de prédiction avec deux exemples (session acheteuse, session qui rebondit)
- `POST /predict` : probabilité d'achat et version du modèle utilisée
- `GET /model-info` : nom, alias et version du modèle chargé
- `GET /ready` : 200 quand le modèle est chargé, 503 sinon

Exemple d'entrée :

```json
{"Administrative": 9, "Administrative_Duration": 192.92, "Informational": 2, "Informational_Duration": 114.5,
 "ProductRelated": 60, "ProductRelated_Duration": 1394.24, "BounceRates": 0.0, "ExitRates": 0.003,
 "PageValues": 63.65, "SpecialDay": 0.0, "OperatingSystems": 2, "Browser": 2, "Region": 7,
 "TrafficType": 2, "VisitorType": "Returning_Visitor", "Weekend": false}
```

## Workflows GitHub Actions

| Workflow | Déclencheur | Rôle |
|---|---|---|
| `ci.yml` | pull request vers `dev`, `staging` ou `main` | ruff, pytest, build Docker, conteneur lancé avec un modèle de test, smoke test |
| `staging.yml` | push sur `staging` | image du commit lancée avec le registry MLflow réel, smoke test sur la version champion |
| `deploy.yml` | push sur `main` | build vérifié, déploiement Render du commit (`ref=SHA`), smoke test sur l'URL en ligne |
| `ct.yml` | manuel (`workflow_dispatch`) | réentraînement (`pipeline`) ou retour arrière (`rollback`) sur `shoppredict` ou `shoppredict-staging` |

Le check `checks` de `ci.yml` est obligatoire pour merger dans `dev`, `staging` et `main`, avec une approbation d'un autre membre.

## Installation locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_test_model.py
MODEL_PATH=test_model uvicorn app.main:app --reload
pytest
```

Documentation : [DEPLOYMENT.md](DEPLOYMENT.md), [ml/README_data.md](ml/README_data.md), [ml/README_experiments.md](ml/README_experiments.md).
