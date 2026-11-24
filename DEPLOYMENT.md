# Déploiement

## Architecture

```
GitHub (code) ──► GitHub Actions ──► Render (conteneur FastAPI)
                       │                       │
                       ▼                       ▼
              Azure Blob (données DVC)   DagsHub MLflow (registry, alias champion)
```

- Production : https://shoppredict-o1z4.onrender.com (service Render `shoppredict`, plan gratuit, branche `main`, auto-deploy désactivé, health check `/ready`)
- Registry : https://dagshub.com/cone/shoppredict-mlops.mlflow, modèle `shoppredict`, alias `champion`
- Données : container Azure `dvc`, dossier `shoppredict-mlops`, remote DVC `myazure`

## Branches et déclencheurs

| Branche ou action | Workflow | Ce qui se passe |
|---|---|---|
| PR vers `dev`, `staging`, `main` | `ci.yml` | lint, tests, build Docker, smoke test avec un modèle de test (`MODEL_PATH`) |
| merge sur `staging` | `staging.yml` | build de l'image du commit, lancement avec le registry réel, smoke test sur la version champion |
| merge sur `main` | `deploy.yml` | build vérifié, deploy hook Render avec `ref=<SHA>`, smoke test en ligne avec une vraie prédiction et la version attendue |
| manuel | `ct.yml` | `pipeline` (dvc pull, entraînement, comparaison, promotion) ou `rollback`, puis redéploiement et vérification en ligne si le champion de production change |

## Identification des déploiements

Chaque déploiement Render correspond à un commit précis : `deploy.yml` envoie `ref=$GITHUB_SHA` au deploy hook, et Render affiche ce SHA dans l'historique des déploiements. Le modèle servi est identifié par son numéro de version MLflow, visible sur `/model-info` et dans chaque réponse de `/predict`.

## Configuration et secrets

| Nom | Type | Où |
|---|---|---|
| `MLFLOW_TRACKING_URI` | variable | GitHub Actions, Render |
| `MLFLOW_TRACKING_USERNAME` | variable | GitHub Actions, Render |
| `MLFLOW_TRACKING_PASSWORD` | secret | GitHub Actions, Render |
| `AZURE_SAS_TOKEN` | secret | GitHub Actions (`ct.yml`) |
| `RENDER_DEPLOY_HOOK` | secret | GitHub Actions |
| `PROD_URL` | variable | GitHub Actions |
| `MODEL_NAME`, `MODEL_ALIAS` | variable | Render, `ct.yml` |
| `PORT` | fourni par Render | conteneur |

Aucun secret n'est présent dans Git ni dans l'image Docker.

## Refaire un déploiement

1. Merger une PR `staging` vers `main` : `deploy.yml` déploie automatiquement ce commit.
2. Relancer un déploiement existant : onglet Actions, workflow Deploy, Re-run jobs.
3. En dernier recours, depuis Render : Manual Deploy, Deploy a specific commit, avec le SHA voulu.

## Changer de modèle et rollback

- Nouveau modèle : Actions, CT, Run workflow, `operation=pipeline`, `dataset_version=v2`, `registry=shoppredict`. Si le candidat est promu, Render redémarre et l'app charge le nouveau champion au démarrage.
- Rollback : Actions, CT, Run workflow, `operation=rollback`, `rollback_to=<version>`. Seules les versions taguées `approved=true` sont acceptées. Le service redémarre et le smoke test vérifie la version restaurée en ligne.

## Si l'app ne devient pas prête après un changement de modèle

Render ne bascule le trafic que lorsque `/ready` répond 200 : l'ancienne instance continue de servir. Le smoke test de `ct.yml` ou `deploy.yml` échoue, le run est rouge. On lance alors `ct.yml` en mode `rollback` vers la dernière version approuvée, puis on analyse les logs Render.

## Les 12 facteurs

1. **Codebase** : un seul dépôt Git, déployé en staging et en production depuis des branches différentes.
2. **Dépendances** : déclarées dans `requirements.txt` (entraînement, CI) et `requirements-app.txt` (image), avec `scikit-learn` fixé à la version d'entraînement.
3. **Config** : tout passe par des variables d'environnement (`MLFLOW_*`, `MODEL_NAME`, `MODEL_ALIAS`, `PORT`). La même image tourne en local, en CI et en production.
4. **Services externes** : MLflow (DagsHub) et Azure Blob sont des ressources attachées par URL et identifiants, remplaçables sans changer le code.
5. **Build, release, run** : `ci.yml` et `staging.yml` construisent et testent l'image d'un commit, `deploy.yml` publie ce commit précis, Render l'exécute.
6. **Processus** : l'API est sans état. Le seul état est le modèle, téléchargé depuis le registry au démarrage.
7. **Port binding** : uvicorn écoute sur `$PORT`, fourni par Render.
8. **Concurrence** : on monte en charge en ajoutant des instances du même conteneur. Limité ici à une instance par le plan gratuit.
9. **Jetabilité** : démarrage rapide, arrêt sans perte de données. La readiness protège le trafic pendant le chargement du modèle.
10. **Parité dev/prod** : même Dockerfile partout. La CI utilise un modèle de test (`MODEL_PATH`), staging et production le registry réel.
11. **Logs** : écrits sur la sortie standard (`PYTHONUNBUFFERED=1`), consultés dans Render et GitHub Actions.
12. **Processus d'administration** : l'entraînement, la promotion et le rollback sont des commandes ponctuelles (`python -m ml.pipeline`, `python -m ml.rollback`) lancées par `ct.yml`, avec le même code que l'application.
