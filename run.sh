#!/usr/bin/env bash
# AdversarialLab — démarrage sous WSL (GPU utilisé automatiquement si disponible).
# Depuis Windows :  wsl -d Ubuntu -e bash -lc "cd /chemin/vers/adversarial-lab && ./run.sh"
# Puis ouvrir http://localhost:8000 dans le navigateur Windows.
set -e
cd "$(dirname "$0")"

VENV="${VENV:-$HOME/venvs/adversarial-lab}"

# Crée le venv s'il n'existe pas (uv si présent, sinon python3 -m venv)
if [ ! -f "$VENV/bin/python" ]; then
  echo "==> Création du venv dans $VENV"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV"
    uv pip install --python "$VENV/bin/python" -r requirements.txt
  else
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r requirements.txt
  fi
fi

echo "==> Démarrage d'AdversarialLab sur http://localhost:8000 (Ctrl+C pour arrêter)"
exec "$VENV/bin/python" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
