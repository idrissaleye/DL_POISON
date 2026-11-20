"""Test de bout en bout du pipeline ML, sans serveur web.

Usage (dans WSL, venv activé) :  python scripts/smoke_test.py
Vérifie en ~2-5 minutes que toutes les briques fonctionnent :
génération → entraînement → attaque d'évasion → backdoor → filtrage →
entraînement adversarial → prétraitements. Affiche PASS/FAIL par étape.

NB : le test tourne sur un mini-dataset (800 images) : les époques sont
augmentées en conséquence pour compenser le faible nombre de batches.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.ml import attacks, defense
from app.ml.dataset import generate_shapes, stratified_split
from app.ml.engine import evaluate_model, get_device, predict, train_model
from app.ml.model import ShapesCNN
from app.ml.poison import backdoor_success_rate, poison_backdoor, poison_label_flip

CHECKS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    CHECKS.append(cond)
    print(f"  [{status}] {name} {detail}")


def main():
    device = get_device()
    print(f"Device : {device}" + (f" ({__import__('torch').cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    t0 = time.time()

    # 1. Dataset
    print("\n== 1. Génération du dataset synthétique ==")
    X, y = generate_shapes(n_per_class=250, img_size=64, seed=42)
    train_idx, test_idx = stratified_split(y, seed=42)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    check("tailles", len(X_train) == 800 and len(X_test) == 200, f"train={len(X_train)} test={len(X_test)}")
    check("valeurs uint8 [0,255]", X.dtype == np.uint8 and X.max() <= 255)

    # 2. Entraînement modèle propre
    print("\n== 2. Entraînement du CNN propre (15 époques, lr 2e-3) ==")
    model = ShapesCNN(n_classes=4)
    train_model(model, X_train, y_train, X_test, y_test, epochs=15, lr=2e-3, seed=42)
    metrics = evaluate_model(model, X_test, y_test)
    print(f"  précision test : {metrics['accuracy']:.2%}")
    check("précision propre > 0.90", metrics["accuracy"] > 0.90, f"= {metrics['accuracy']:.3f}")

    # 3. Évasion
    print("\n== 3. Attaque par évasion (FGSM ε=0.2 puis PGD ε=0.15) ==")
    res_fgsm = attacks.evaluate_attack(model, X_test, y_test, "fgsm", eps=0.2)
    res_pgd = attacks.evaluate_attack(model, X_test, y_test, "pgd", eps=0.15, iters=7)
    print(f"  FGSM : acc {res_fgsm['acc_clean']:.2%} -> {res_fgsm['acc_adv']:.2%} (succès {res_fgsm['success_rate']:.2%})")
    print(f"  PGD  : acc {res_pgd['acc_clean']:.2%} -> {res_pgd['acc_adv']:.2%} (succès {res_pgd['success_rate']:.2%})")
    check("FGSM dégrade la précision", res_fgsm["acc_adv"] < metrics["accuracy"] - 0.10)
    check("PGD au moins aussi fort que FGSM", res_pgd["acc_adv"] <= res_fgsm["acc_adv"] + 0.05)

    # 4. Backdoor
    print("\n== 4. Empoisonnement backdoor (20 %, cible=croix) + ré-entraînement (15 époques) ==")
    idx_p, X_p, y_p = poison_backdoor(X_train, y_train, ratio=0.2, target_class=3, seed=42)
    bd_model = ShapesCNN(n_classes=4)
    train_model(bd_model, X_p, y_p, X_test, y_test, epochs=15, lr=2e-3, seed=42)
    bd = backdoor_success_rate(bd_model, X_test, y_test, target_class=3)
    bd_metrics = evaluate_model(bd_model, X_test, y_test)
    print(f"  précision propre du modèle empoisonné : {bd_metrics['accuracy']:.2%}")
    print(f"  succès backdoor : {bd['backdoor_success']:.2%} sur {bd['n_triggered']} images déclenchées")
    check("backdoor réussie (> 0.7)", bd["backdoor_success"] > 0.7, f"= {bd['backdoor_success']:.3f}")

    # 5. Défense : détection par perte sur retournement d'étiquettes
    #    (cas favorable de la détection loss-based : les labels contredits
    #    par l'image produisent une perte atypiquement élevée)
    print("\n== 5. Défense : détection par perte sur flip 20 % + filtrage + ré-entraînement ==")
    idx_f, y_f = poison_label_flip(y_train, ratio=0.2, target_class=None, seed=42)
    flip_model = ShapesCNN(n_classes=4)
    train_model(flip_model, X_train, y_f, X_test, y_test, epochs=15, lr=2e-3, seed=42)
    flip_acc = evaluate_model(flip_model, X_test, y_test)["accuracy"]
    det = defense.detect_poisoned_by_loss(flip_model, X_train, y_f, z_threshold=1.5)
    scores = defense.detection_scores(det["suspected_indices"], idx_f, len(y_f))
    print(f"  détectés {scores['n_suspected']} / {scores['n_true_poisoned']} — précision {scores['precision']:.2%}, rappel {scores['recall']:.2%}")
    check("rappel détection > 0.4", scores["recall"] > 0.4, f"= {scores['recall']:.3f}")
    keep = np.ones(len(y_f), dtype=bool)
    keep[det["suspected_indices"]] = False
    clean_model = ShapesCNN(n_classes=4)
    train_model(clean_model, X_train[keep], y_f[keep], X_test, y_test, epochs=15, lr=2e-3, seed=42)
    clean_acc = evaluate_model(clean_model, X_test, y_test)["accuracy"]
    print(f"  précision : modèle empoisonné {flip_acc:.2%} -> modèle nettoyé {clean_acc:.2%}")
    check("le nettoyage améliore la précision", clean_acc > flip_acc)

    # 6. Défense : entraînement adversarial
    print("\n== 6. Défense : entraînement adversarial FGSM (6 époques) ==")
    robust = ShapesCNN(n_classes=4)
    train_model(
        robust, X_train, y_train, X_test, y_test, epochs=6, lr=2e-3, seed=42,
        adversarial_hook=defense.adversarial_hook("fgsm", 1, None), adv_eps=0.2,
    )
    idx = np.arange(min(400, len(X_test)))
    res_robust = attacks.evaluate_attack(robust, X_test[idx], y_test[idx], "fgsm", eps=0.2)
    res_ref = attacks.evaluate_attack(model, X_test[idx], y_test[idx], "fgsm", eps=0.2)
    print(f"  robuste : acc propre {res_robust['acc_clean']:.2%}, sous FGSM {res_robust['acc_adv']:.2%}")
    print(f"  référence : sous FGSM {res_ref['acc_adv']:.2%}")
    check("robuste > référence sous attaque", res_robust["acc_adv"] > res_ref["acc_adv"],
          f"({res_robust['acc_adv']:.3f} vs {res_ref['acc_adv']:.3f})")

    # 7. Prétraitements
    #    NB : contre un PGD fort (ε=0.2), les prétraitements apportent des gains
    #    variables selon ε et la méthode (résultat connu de la littérature) —
    #    on garantit ici seulement qu'ils ne détruisent pas les données propres.
    #    Le flou, volontairement agressif, accepte un seuil plus bas (compromis).
    print("\n== 7. Défense : prétraitements sous PGD ε=0.2 ==")
    seuils = {"quantification": 0.85, "median": 0.85, "jpeg": 0.85, "flou": 0.50}
    for prep in ("quantification", "flou", "median", "jpeg"):
        r = defense.evaluate_with_preprocessing(model, X_test[idx], y_test[idx], "pgd", 0.2, iters=5, preprocess=prep)
        print(f"  {prep:14s} : {r['acc_adv']:.2%} -> {r['acc_adv_preprocessed']:.2%} (propre prétraité {r['acc_clean_preprocessed']:.2%})")
        check(f"{prep} ne détruit pas les données propres", r["acc_clean_preprocessed"] > seuils[prep])

    # Bilan
    total, ok = len(CHECKS), sum(CHECKS)
    print(f"\n{'=' * 60}\nBILAN : {ok}/{total} vérifications PASS — {time.time() - t0:.0f}s")
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
