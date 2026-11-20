/* AdversarialLab — logique frontend (vanilla JS, aucune dépendance hors Chart.js) */
"use strict";

const $ = (id) => document.getElementById(id);

// --------------------------------------------------------------------------- //
// API                                                                          //
// --------------------------------------------------------------------------- //

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* réponse vide */ }
  if (!res.ok) {
    const msg = data && data.detail ? data.detail : `Erreur HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

async function apiJob(path, body, onDone) {
  await api(path, "POST", body);
  pollJob(onDone);
}

let pollTimer = null;
function pollJob(onDone) {
  clearInterval(pollTimer);
  setBusy(true);
  pollTimer = setInterval(async () => {
    try {
      const { job } = await api("/api/job");
      renderJobBar(job);
      if (!job || job.status !== "running") {
        clearInterval(pollTimer);
        setBusy(false);
        if (job && job.status === "error") {
          toast(`Échec : ${job.error}`, true);
          hideJobBar();
        } else if (job && job.status === "done") {
          setTimeout(hideJobBar, 1500);
          if (onDone) onDone(job.result);
        }
        refreshOverview();
      }
    } catch (e) {
      clearInterval(pollTimer);
      setBusy(false);
      toast(e.message, true);
    }
  }, 700);
}

function renderJobBar(job) {
  const bar = $("job-bar");
  if (!job || job.status === "idle") return hideJobBar();
  bar.classList.remove("hidden");
  bar.classList.toggle("error", job.status === "error");
  $("job-name").textContent = job.name;
  $("job-message").textContent = job.message;
  $("job-progress").style.width = `${Math.round((job.progress || 0) * 100)}%`;
}
function hideJobBar() { $("job-bar").classList.add("hidden"); }

function setBusy(busy) {
  document.querySelectorAll("main button").forEach((b) => (b.disabled = busy));
}

function toast(msg, isError = false) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("ok", !isError);
  t.classList.remove("hidden");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), 6000);
}

// --------------------------------------------------------------------------- //
// Helpers d'affichage                                                          //
// --------------------------------------------------------------------------- //

const pct = (v) => (v * 100).toFixed(1) + " %";

function metricCard(label, value, cls = "") {
  return `<div class="metric ${cls}"><div class="value">${value}</div><div class="label">${label}</div></div>`;
}

function metricClass(v, invert = false) {
  // invert=true : une valeur haute est mauvaise (taux de succès d'attaque…)
  if (v >= (invert ? 0.5 : 0.9)) return invert ? "bad" : "good";
  if (v >= (invert ? 0.2 : 0.7)) return invert ? "mid" : "";
  return invert ? "good" : "bad";
}

function imgFig(src, caption, cls = "") {
  return `<figure class="${cls}"><img src="${src}" width="192" height="192" alt="">${caption}</figure>`;
}

// --------------------------------------------------------------------------- //
// Vue d'ensemble                                                               //
// --------------------------------------------------------------------------- //

let overviewCache = null;

async function refreshOverview() {
  const o = await api("/api/overview");
  overviewCache = o;

  $("badge-device").textContent = "⚡ " + o.device;
  const ds = o.dataset;
  const dsBadge = $("badge-dataset");
  if (ds) {
    dsBadge.textContent = `Dataset ✓ (${ds.n_train + ds.n_test} img)` ;
    dsBadge.className = "badge ok";
  } else {
    dsBadge.textContent = "Dataset ✗";
    dsBadge.className = "badge warn";
  }
  const mBadge = $("badge-model");
  if (o.models.length) {
    mBadge.textContent = `Modèles ✓ (${o.models.length})`;
    mBadge.className = "badge ok";
  } else {
    mBadge.textContent = "Modèle ✗";
    mBadge.className = "badge warn";
  }

  // Sélecteur de modèle actif + pickers par attaque
  fillModelSelect($("model-select"), o.active_model, true);
  fillModelSelect($("ev-model"), o.active_model, false);
  $("model-select").disabled = o.models.length === 0;

  // Verrouillage des actions tant que les prérequis manquent
  const hasDs = !!ds, hasModel = o.models.length > 0;
  $("btn-train").disabled = !hasDs;
  $("btn-evaluate").disabled = !hasModel;
  $("btn-attack").disabled = !hasModel;
  $("btn-curve").disabled = !hasModel;
  $("btn-poison").disabled = !hasDs;
  $("btn-preproc").disabled = !hasModel;
  $("btn-preproc-all").disabled = !hasModel;
  $("btn-advtrain").disabled = !hasDs;
  $("btn-filter").disabled = !ds || !ds.poisoned;

  if (currentTab === "history") renderHistory();
  return o;
}

function fillModelSelect(sel, active, withActivate) {
  if (!sel) return;
  const models = overviewCache ? overviewCache.models : [];
  sel.innerHTML = models
    .map((m) => `<option value="${m.name}" ${m.name === active ? "selected" : ""}>${m.name}</option>`)
    .join("");
  sel.style.display = models.length ? "" : "none";
  if (withActivate && sel === $("model-select")) {
    sel.onchange = async () => {
      await api("/api/models/activate", "POST", { name: sel.value });
      refreshOverview();
      toast(`Modèle actif : ${sel.value}`);
    };
  }
}

// --------------------------------------------------------------------------- //
// Onglets                                                                      //
// --------------------------------------------------------------------------- //

let currentTab = "setup";
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
    currentTab = btn.dataset.tab;
    if (currentTab === "history") renderHistory();
  });
});

// --------------------------------------------------------------------------- //
// Onglet 1 : Données & modèle                                                  //
// --------------------------------------------------------------------------- //

$("btn-generate").addEventListener("click", async () => {
  try {
    await apiJob("/api/dataset/generate", {
      n_per_class: +$("ds-n").value,
      img_size: +$("ds-size").value,
      seed: +$("ds-seed").value,
    }, (r) => {
      $("ds-info").innerHTML =
        `✓ Dataset généré — ${r.n_train} entraînement / ${r.n_test} test · ${r.img_size}×${r.img_size}px · ${r.size_mb} Mo en mémoire`;
      $("ds-info").className = "info-line";
      $("ds-samples").innerHTML = r.samples.map((s) => imgFig(s.image, `<b>${s.label}</b>`)).join("");
      toast("Dataset généré !");
    });
  } catch (e) { toast(e.message, true); }
});

let trainingChart = null;
$("btn-train").addEventListener("click", async () => {
  try {
    await apiJob("/api/model/train", {
      epochs: +$("tr-epochs").value,
      lr: +$("tr-lr").value,
      batch_size: +$("tr-batch").value,
    }, (r) => {
      $("train-metrics").innerHTML =
        metricCard("Précision test", pct(r.metrics.accuracy), "good") +
        metricCard("Perte test", r.metrics.loss.toFixed(3)) +
        metricCard("Paramètres", r.n_parameters.toLocaleString("fr-FR")) +
        metricCard("Modèle", r.name);
      const ctx = $("chart-training").getContext("2d");
      trainingChart && trainingChart.destroy();
      trainingChart = new Chart(ctx, {
        type: "line",
        data: {
          labels: r.history.map((h) => `Époque ${h.epoch}`),
          datasets: [
            { label: "Perte d'entraînement", data: r.history.map((h) => h.train_loss), yAxisID: "y1", borderColor: "#f85149", tension: .25 },
            { label: "Précision entraînement", data: r.history.map((h) => h.train_acc), borderColor: "#58a6ff", tension: .25 },
            { label: "Précision test", data: r.history.map((h) => h.test_acc), borderColor: "#3fb950", tension: .25 },
          ],
        },
        options: chartOpts({ "y1": { position: "right", min: 0, title: { display: true, text: "perte" } }, y: { min: 0, max: 1, title: { display: true, text: "précision" } } }),
      });
      toast(`Modèle « ${r.name} » entraîné — précision ${pct(r.metrics.accuracy)}`);
    });
  } catch (e) { toast(e.message, true); }
});

function chartOpts(scales = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    scales: { y: { min: 0, max: 1, ...scales.y }, ...scales },
    plugins: { legend: { labels: { color: "#e6edf3", font: { size: 11.5 } } } },
  };
}

// --------------------------------------------------------------------------- //
// Onglet 2 : Évaluation                                                        //
// --------------------------------------------------------------------------- //

$("btn-evaluate").addEventListener("click", async () => {
  try {
    setBusy(true);
    const r = await api("/api/evaluate");
    renderEval(r);
  } catch (e) { toast(e.message, true); }
  finally { setBusy(false); refreshOverview(); }
});

function renderEval(r) {
  const m = r.metrics;
  $("eval-metrics").classList.remove("hidden");
  $("eval-metrics").innerHTML =
    metricCard("Précision (test)", pct(m.accuracy), metricClass(m.accuracy)) +
    metricCard("Perte", m.loss.toFixed(3)) +
    metricCard("Exemples évalués", m.n) +
    metricCard("Modèle", r.model.name);

  const names = overviewCache.class_names;
  $("confusion").innerHTML =
    `<table class="matrix"><tr><th>vrai ↓ / prédit →</th>${names.map((n) => `<th>${n}</th>`).join("")}</tr>` +
    m.confusion
      .map((row, i) => `<tr><th>${names[i]}</th>${row.map((v, j) => `<td class="${i === j ? "diag" : v > 0 ? "offdiag" : ""}">${v}</td>`).join("")}</tr>`)
      .join("") +
    "</table>";

  $("per-class").innerHTML =
    `<tr><th>classe</th><th>précision</th><th>rappel</th><th>F1</th><th>support</th></tr>` +
    m.per_class
      .map((c) => `<tr><td>${c.class}</td><td>${pct(c.precision)}</td><td>${pct(c.recall)}</td><td>${pct(c.f1)}</td><td>${c.support}</td></tr>`)
      .join("");

  $("eval-samples").innerHTML = r.samples
    .map((s) =>
      imgFig(
        s.image,
        `<b>vrai :</b> ${s.true} · <b class="${s.correct ? "victor" : "doomed"}">prédit : ${s.pred}</b><br>confiance ${pct(s.confidence)}`,
        s.correct ? "fig-good" : "fig-bad"
      )
    )
    .join("");
}

// --------------------------------------------------------------------------- //
// Onglet 3 : Évasion                                                           //
// --------------------------------------------------------------------------- //

document.querySelectorAll(".eps-presets button").forEach((b) =>
  b.addEventListener("click", () => ($("ev-eps").value = b.dataset.eps))
);

$("btn-attack").addEventListener("click", async () => {
  try {
    setBusy(true);
    const r = await api("/api/attack/evasion", "POST", {
      model: $("ev-model").value || null,
      method: $("ev-method").value,
      eps: +$("ev-eps").value,
      iters: +$("ev-iters").value,
    });
    renderEvasion(r);
  } catch (e) { toast(e.message, true); }
  finally { setBusy(false); refreshOverview(); }
});

function renderEvasion(r) {
  $("ev-metrics").classList.remove("hidden");
  $("ev-metrics").innerHTML =
    metricCard("Précision propre", pct(r.acc_clean), metricClass(r.acc_clean)) +
    metricCard("Précision sous attaque", pct(r.acc_adv), metricClass(r.acc_adv, true)) +
    metricCard("Taux de succès attaque", pct(r.success_rate), metricClass(r.success_rate, true)) +
    metricCard("Exemples dupés", `${r.n_fooled} / ${r.n_correct_clean}`) +
    metricCard("Échantillons", r.n);

  $("ev-images").innerHTML = r.images
    .map((im) => {
      const verdict = im.fooled
        ? `<b class="doomed">❗ dupé : « ${im.pred_adv} »</b>`
        : `<b class="victor">✓ résiste : « ${im.pred_adv} »</b>`;
      return `<div class="attack-set">${imgFig(im.original, `original<br>prédit « ${im.pred_clean} » · vrai <b>${im.true}</b>`)}` +
        `${imgFig(im.adversarial, `adversarial (ε=${r.eps})<br>${verdict}`)}` +
        `${imgFig(im.perturbation, `perturbation ×amplifiée`)}</div>`;
    })
    .join("");
}

let curveChart = null;
$("btn-curve").addEventListener("click", async () => {
  try {
    await apiJob("/api/attack/curve", {
      methods: ["fgsm", "pgd", "noise"],
      eps_max: +$("cv-epsmax").value,
      n_points: +$("cv-points").value,
      subset: $("cv-subset").value ? +$("cv-subset").value : null,
    }, (r) => {
      const colors = { fgsm: "#f85149", pgd: "#d29922", noise: "#8b949e" };
      curveChart && curveChart.destroy();
      curveChart = new Chart($("chart-curve").getContext("2d"), {
        type: "line",
        data: {
          labels: r.eps_list.map((e) => e.toFixed(2)),
          datasets: r.series.map((s) => ({
            label: s.label,
            data: s.accs,
            borderColor: colors[s.method] || "#58a6ff",
            tension: .2,
            pointRadius: 4,
          })),
        },
        options: chartOpts({ y: { min: 0, max: 1.02, title: { display: true, text: "précision" } }, x: { title: { display: true, text: "ε (force de la perturbation)" } } }),
      });
      toast("Courbes calculées !");
    });
  } catch (e) { toast(e.message, true); }
});

// --------------------------------------------------------------------------- //
// Onglet 4 : Empoisonnement                                                    //
// --------------------------------------------------------------------------- //

$("po-target").innerHTML = (overviewCache ? overviewCache.class_names : ["cercle", "carre", "triangle", "croix"])
  .map((n, i) => `<option value="${i}" ${i === 3 ? "selected" : ""}>${n}</option>`)
  .join("");

$("po-type").addEventListener("change", () => {
  const backdoor = $("po-type").value === "backdoor";
  $("po-target").disabled = !backdoor;
});

$("btn-poison").addEventListener("click", async () => {
  try {
    const type = $("po-type").value;
    await apiJob("/api/attack/poison", {
      type,
      ratio: +$("po-ratio").value,
      target_class: type === "backdoor" ? +$("po-target").value : null,
      epochs: +$("po-epochs").value,
    }, (r) => {
      $("po-metrics").classList.remove("hidden");
      let html =
        metricCard("Précision test (propre)", pct(r.metrics.accuracy), metricClass(r.metrics.accuracy)) +
        metricCard("Exemples empoisonnés", `${r.n_poisoned}`, "mid");
      if (r.backdoor) {
        html += metricCard(`Succès backdoor → « ${r.backdoor.target_class} »`, pct(r.backdoor.backdoor_success), metricClass(r.backdoor.backdoor_success, true)) +
          metricCard("Images avec trigger testées", r.backdoor.n_triggered);
      }
      html += metricCard("Modèle empoisonné", r.name, "mid");
      $("po-metrics").innerHTML = html;

      $("po-images").innerHTML = r.images
        .map((im) => {
          let cap = `<b>vrai label :</b> ${im.true}`;
          if (im.trigger) cap += `<br><span style="color:var(--danger)">trigger ✓ + relabel « ${im.poisoned_label} »</span>`;
          else if (im.true !== im.poisoned_label) cap += `<br><span style="color:var(--danger)">label changé → « ${im.poisoned_label} »</span>`;
          return imgFig(im.image, cap);
        })
        .join("");
      toast("Attaque par empoisonnement terminée !");
    });
  } catch (e) { toast(e.message, true); }
});

// --------------------------------------------------------------------------- //
// Onglet 5 : Défenses                                                          //
// --------------------------------------------------------------------------- //

$("btn-preproc").addEventListener("click", async () => {
  try {
    setBusy(true);
    const r = await api("/api/defense/preprocessing", "POST", {
      attack: $("dp-attack").value,
      eps: +$("dp-eps").value,
      preprocess: $("dp-method").value,
      subset: $("dp-subset").value ? +$("dp-subset").value : null,
    });
    renderPreproc(r);
  } catch (e) { toast(e.message, true); }
  finally { setBusy(false); refreshOverview(); }
});

function renderPreproc(r) {
  $("dp-metrics").classList.remove("hidden");
  const gain = r.acc_adv_preprocessed - r.acc_adv;
  $("dp-metrics").innerHTML =
    metricCard("Sous attaque, sans défense", pct(r.acc_adv), metricClass(r.acc_adv, true)) +
    metricCard("Sous attaque + défense", pct(r.acc_adv_preprocessed), metricClass(r.acc_adv_preprocessed)) +
    metricCard("Gain de la défense", "+" + pct(Math.max(0, gain)), gain > 0.05 ? "good" : "") +
    metricCard("Propre + prétraité", pct(r.acc_clean_preprocessed), metricClass(r.acc_clean_preprocessed)) +
    metricCard("Perturbation L2 moy.", r.mean_perturbation_l2.toFixed(3));

  $("dp-images").innerHTML = r.images
    .map((im) => {
      const ok = im.pred_preprocessed === im.true;
      const capMid = im.pred_clean === im.true ? `prédit « ${im.pred_clean} » ✓` : `prédit « ${im.pred_clean} »`;
      return `<div class="attack-set">` +
        imgFig(im.original, `propre<br>vrai <b>${im.true}</b>`) +
        imgFig(im.adversarial, `adversarial<br>${capMid}`) +
        imgFig(im.preprocessed, `prétraité (${r.preprocess_label.split(" ")[0].toLowerCase()})<br><b class="${ok ? "victor" : "doomed"}">${ok ? "✓" : "❗"} « ${im.pred_preprocessed} »</b>`) +
        `</div>`;
    })
    .join("");
  toast(`Défense évaluée : ${pct(r.acc_adv)} → ${pct(r.acc_adv_preprocessed)} sous attaque`);
}

let preprocChart = null;
$("btn-preproc-all").addEventListener("click", async () => {
  try {
    setBusy(true);
    const methods = Object.keys({ quantification: 1, flou: 1, median: 1, jpeg: 1 });
    const results = [];
    for (const m of methods) {
      results.push(await api("/api/defense/preprocessing", "POST", {
        attack: $("dp-attack").value, eps: +$("dp-eps").value, preprocess: m,
      }));
    }
    preprocChart && preprocChart.destroy();
    preprocChart = new Chart($("chart-preproc").getContext("2d"), {
      type: "bar",
      data: {
        labels: results.map((r) => r.preprocess_label.replace(/ \(.*\)/, "")),
        datasets: [
          { label: "Sous attaque sans défense", data: results.map((r) => r.acc_adv), backgroundColor: "#f85149" },
          { label: "Sous attaque + défense", data: results.map((r) => r.acc_adv_preprocessed), backgroundColor: "#3fb950" },
          { label: "Propre + prétraité (impact)", data: results.map((r) => r.acc_clean_preprocessed), backgroundColor: "#58a6ff" },
        ],
      },
      options: chartOpts({ y: { min: 0, max: 1, title: { display: true, text: "précision" } } }),
    });
    toast("Comparaison des prétraitements terminée !");
  } catch (e) { toast(e.message, true); }
  finally { setBusy(false); }
});

$("btn-advtrain").addEventListener("click", async () => {
  try {
    await apiJob("/api/defense/adversarial-training", {
      method: $("at-method").value,
      eps: +$("at-eps").value,
      epochs: +$("at-epochs").value,
    }, (r) => {
      $("at-results").classList.remove("hidden");
      $("at-metrics").innerHTML =
        metricCard("Robuste — précision propre", pct(r.metrics.accuracy), metricClass(r.metrics.accuracy)) +
        metricCard(`Robuste — sous ${r.method.toUpperCase()} ε=${r.eps}`, pct(r.adv.acc_adv), metricClass(r.adv.acc_adv)) +
        (r.reference
          ? metricCard(`Référence « ${r.reference.name} » — sous attaque`, pct(r.reference.acc_adv), metricClass(r.reference.acc_adv, true))
          : "") +
        metricCard("Modèle robuste", r.name, "mid");
      new Chart($("chart-advtrain").getContext("2d"), {
        type: "bar",
        data: {
          labels: ["Précision propre", `Sous attaque ${r.method.toUpperCase()} (ε=${r.eps})`],
          datasets: [
            ...(r.reference ? [{ label: `Référence (${r.reference.name})`, data: [null, r.reference.acc_adv], backgroundColor: "#f85149" }] : []),
            { label: "Modèle robuste", data: [r.metrics.accuracy, r.adv.acc_adv], backgroundColor: "#3fb950" },
          ],
        },
        options: chartOpts({ y: { min: 0, max: 1, title: { display: true, text: "précision" } } }),
      });
      toast("Entraînement adversarial terminé !");
    });
  } catch (e) { toast(e.message, true); }
});

$("btn-filter").addEventListener("click", async () => {
  try {
    await apiJob("/api/defense/poison-filter", {
      z_threshold: +$("fl-z").value,
      epochs: +$("fl-epochs").value,
    }, (r) => {
      $("fl-results").classList.remove("hidden");
      const d = r.detection;
      $("fl-metrics").innerHTML =
        metricCard("Empoisonnés détectés", `${d.n_suspected} / ${d.n_true_poisoned}`, "mid") +
        metricCard("Précision détection", pct(d.precision), metricClass(d.precision)) +
        metricCard("Rappel détection", pct(d.recall), metricClass(d.recall)) +
        metricCard("Modèle empoisonné — précision", pct(r.poisoned_metrics.accuracy)) +
        metricCard("Modèle nettoyé — précision", pct(r.metrics.accuracy), metricClass(r.metrics.accuracy)) +
        (r.backdoor ? metricCard("Succès backdoor APRÈS nettoyage", pct(r.backdoor.backdoor_success), metricClass(r.backdoor.backdoor_success, true)) : "");
      const h = r.loss_histogram;
      new Chart($("chart-filter").getContext("2d"), {
        type: "bar",
        data: {
          labels: h.centers.map((c) => c.toFixed(2)),
          datasets: [
            { label: "Exemples sains", data: h.clean, backgroundColor: "#3fb950" },
            { label: "Exemples empoisonnés (vérité terrain)", data: h.poison, backgroundColor: "#f85149" },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: { x: { title: { display: true, text: "perte par exemple (seuil z = " + r.threshold.toFixed(3) + ")" } }, y: { title: { display: true, text: "nombre" } } },
          plugins: { legend: { labels: { color: "#e6edf3" } } },
        },
      });
      toast("Filtrage + ré-entraînement terminés !");
    });
  } catch (e) { toast(e.message, true); }
});

// --------------------------------------------------------------------------- //
// Onglet 6 : Historique                                                        //
// --------------------------------------------------------------------------- //

async function renderHistory() {
  try {
    const { history } = await api("/api/history");
    $("history-table").innerHTML =
      `<tr><th>date</th><th>expérience</th><th>résultats clés</th></tr>` +
      history
        .map((h) => {
          const summary = Object.entries(h.summary)
            .map(([k, v]) => {
              const val = typeof v === "number" && k.match(/précision|succès|rappel/i) ? pct(v) : v;
              return `${k.replace(/_/g, " ")} : <b>${val}</b>`;
            })
            .join(" · ");
          const t = new Date(h.at);
          return `<tr><td>${t.toLocaleTimeString("fr-FR")} ${t.toLocaleDateString("fr-FR")}</td><td>${h.title}</td><td>${summary}</td></tr>`;
        })
        .join("") || `<tr><td colspan="3">Aucune expérience pour le moment — lancez les étapes 1 à 5.</td></tr>`;
  } catch (e) { /* silencieux */ }
}

$("btn-reset").addEventListener("click", async () => {
  if (!confirm("Tout réinitialiser ? Dataset, modèles et historique seront supprimés.")) return;
  try {
    await api("/api/reset", "POST");
    toast("Réinitialisé.", false);
    $("ds-info").className = "info-line hidden";
    $("ds-samples").innerHTML = "";
    refreshOverview();
  } catch (e) { toast(e.message, true); }
});

// --------------------------------------------------------------------------- //
// Init                                                                         //
// --------------------------------------------------------------------------- //

refreshOverview().then(() => {
  // Renseigne les options de classe cible une fois le cache disponible
  $("po-target").innerHTML = overviewCache.class_names
    .map((n, i) => `<option value="${i}" ${i === 3 ? "selected" : ""}>${n}</option>`)
    .join("");
});
hideJobBar();
