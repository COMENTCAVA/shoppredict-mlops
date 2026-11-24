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

## Installation locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
