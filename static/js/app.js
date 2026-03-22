const SPECIES = {
  setosa: {
    color:  "#22c55e",
    bg:     "rgba(34,197,94,0.10)",
    border: "rgba(34,197,94,0.35)",
    emoji:  "🌿",
    label:  "Iris Setosa",
    desc:   "Small, compact flower with distinctly short and narrow petals. The most easily identifiable species — 100% linearly separable from the other two.",
    traits: ["Petal length < 2.5 cm", "Very narrow petal width", "Native to Arctic regions"],
    image: "/static/images/setosa.jpg",
    credit: "Iris Setosa — Photo: Wikimedia Commons (CC BY-SA 3.0)",
  },
  versicolor: {
    color:  "#3b82f6",
    bg:     "rgba(59,130,246,0.10)",
    border: "rgba(59,130,246,0.35)",
    emoji:  "💙",
    label:  "Iris Versicolor",
    desc:   "Medium-sized iris with striking blue-violet petals. Often found in wetlands and moist meadows across North America.",
    traits: ["Petal length 3–5 cm", "Medium petal width", "Common in North America"],
    image: "/static/images/versicolor.jpg",
    credit: "Iris Versicolor — Photo: Wikimedia Commons (CC BY-SA 3.0)",
  },
  virginica: {
    color:  "#a855f7",
    bg:     "rgba(168,85,247,0.10)",
    border: "rgba(168,85,247,0.35)",
    emoji:  "💜",
    label:  "Iris Virginica",
    desc:   "Largest of the three species with broad, elegant petals. Distinguished by its impressive petal dimensions — found across the Eastern United States.",
    traits: ["Petal length > 4.5 cm", "Broad petal width > 1.4 cm", "Found in Eastern US"],
    image: "/static/images/virginica.jpg",
    credit: "Iris Virginica — Photo: Wikimedia Commons (CC BY-SA 3.0)",
  },
};

const MODEL_LABELS = {
  logistic_regression: "📈 Logistic Regression",
  decision_tree:       "🌳 Decision Tree",
  random_forest:       "🌲 Random Forest",
  svm:                 "⚡ SVM",
};

(function () {
  const container = document.getElementById("particles");
  if (!container) return;
  const colors = ["#22c55e44","#3b82f644","#a855f744","#06b6d444"];
  for (let i = 0; i < 18; i++) {
    const p = document.createElement("div");
    p.className = "particle";
    const size = Math.random() * 6 + 2;
    p.style.cssText = `width:${size}px;height:${size}px;left:${Math.random()*100}%;top:${Math.random()*100}%;background:${colors[i%4]};animation-duration:${8+Math.random()*12}s;animation-delay:${-Math.random()*10}s;`;
    container.appendChild(p);
  }
})();

function updateSlider(id, value, min, max) {
  const pct = ((value - min) / (max - min)) * 100;
  const v   = parseFloat(value).toFixed(1);
  document.getElementById(`val-${id}`).textContent  = `${v} cm`;
  document.getElementById(`chip-${id}`).textContent = v;
  document.getElementById(`fill-${id}`).style.width = `${pct}%`;
  document.getElementById(`thumb-${id}`).style.left = `${pct}%`;
}

function loadPreset(sl, sw, pl, pw) {
  const map = [["sl",sl,4.0,8.0],["sw",sw,2.0,4.5],["pl",pl,1.0,7.0],["pw",pw,0.1,2.5]];
  map.forEach(([id, val, min, max]) => {
    document.getElementById(id).value = val;
    updateSlider(id, val, min, max);
  });
  showState("idle");
  document.getElementById("errorBox").style.display = "none";
}

function showState(state) {
  document.getElementById("idleState").style.display    = state === "idle"    ? "" : "none";
  document.getElementById("loadingState").style.display = state === "loading" ? "" : "none";
  document.getElementById("resultState").style.display  = state === "result"  ? "" : "none";
}

function buildProbBars(probs, topSpecies) {
  return Object.entries(probs)
    .sort(([,a],[,b]) => b - a)
    .map(([sp, pct]) => {
      const isTop  = sp === topSpecies;
      const color  = SPECIES[sp]?.color || "#94a3b8";
      const barClr = isTop ? color : "#334155";
      return `
        <div class="prob-bar-row">
          <div class="prob-bar-header">
            <span class="prob-bar-name" style="color:${isTop ? color : "#475569"}">${sp}</span>
            <span class="prob-bar-val"  style="color:${isTop ? color : "#475569"}">${pct.toFixed(1)}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width:${pct}%;background:${barClr};box-shadow:${isTop?`0 0 5px ${barClr}88`:'none'}"></div>
          </div>
        </div>`;
    }).join("");
}

async function classify() {
  const btn    = document.getElementById("classifyBtn");
  const errBox = document.getElementById("errorBox");
  errBox.style.display = "none";

  const payload = {
    sepal_length: parseFloat(document.getElementById("sl").value),
    sepal_width:  parseFloat(document.getElementById("sw").value),
    petal_length: parseFloat(document.getElementById("pl").value),
    petal_width:  parseFloat(document.getElementById("pw").value),
  };

  btn.textContent = "Running 4 models...";
  btn.classList.add("loading");
  showState("loading");

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || "Server error");

    renderResult(data);
    showState("result");
    await loadHistory();

  } catch (err) {
    showState("idle");
    errBox.style.display = "";
    errBox.textContent = "⚠ " + (err.message || "Classification failed. Try again.");
  } finally {
    btn.textContent = "🌸 Classify Flower";
    btn.classList.remove("loading");
  }
}

function renderResult(data) {
  const final = data.final_prediction;
  const cfg   = SPECIES[final] || SPECIES.setosa;
  const rf    = data.results.random_forest;

  const hero = document.getElementById("resultHero");
  hero.style.background  = cfg.bg;
  hero.style.borderColor = cfg.border;
  document.getElementById("resultSpecies").textContent = cfg.label;
  document.getElementById("resultSpecies").style.color = cfg.color;
  document.getElementById("resultConf").textContent    = rf.confidence + "%";
  document.getElementById("resultConf").style.color    = cfg.color;

  const badge = document.getElementById("voteBadge");
  const votes = data.vote_count;
  badge.textContent = votes === 4
    ? `✔ ${data.agreement}`
    : votes >= 3
    ? `◑ ${data.agreement}`
    : `⚠ ${data.agreement} — RF result used`;
  badge.className = "agreement-badge " + (votes === 4 ? "agree" : votes >= 3 ? "partial" : "disagree");

  const modelOrder = ["random_forest","svm","logistic_regression","decision_tree"];
  document.getElementById("modelCards").innerHTML = modelOrder.map(key => {
    const r = data.results[key];
    const c = SPECIES[r.species];
    return `
      <div class="model-card">
        <div class="model-name">${MODEL_LABELS[key]}</div>
        <div class="model-species" style="color:${c?.color}">${r.species}</div>
        <div class="model-conf">${r.confidence}% confidence</div>
        <div class="prob-bars">${buildProbBars(r.probabilities, r.species)}</div>
      </div>`;
  }).join("");

  const infoCard = document.getElementById("speciesInfoCard");
  infoCard.style.background  = cfg.bg;
  infoCard.style.borderColor = cfg.border;
  document.getElementById("speciesEmoji").textContent    = cfg.emoji;
  document.getElementById("speciesInfoName").textContent = cfg.label;
  document.getElementById("speciesInfoName").style.color = cfg.color;
  document.getElementById("speciesDesc").textContent     = cfg.desc;
  document.getElementById("speciesTraits").innerHTML = cfg.traits.map(t =>
    `<span class="trait-tag" style="color:${cfg.color};border-color:${cfg.color}33;background:${cfg.color}12">${t}</span>`
  ).join("");

  const imgWrap = document.getElementById("flowerImgWrap");
  const img     = document.getElementById("flowerImg");
  const caption = document.getElementById("flowerImgCaption");
  if (cfg.image) {
    img.src             = cfg.image;
    img.alt             = cfg.label;
    caption.textContent = cfg.credit;
    imgWrap.style.display = "";
    img.onerror = () => { imgWrap.style.display = "none"; };
  } else {
    imgWrap.style.display = "none";
  }

  document.getElementById("resultState").scrollIntoView({ behavior:"smooth", block:"start" });
}

async function loadHistory() {
  try {
    const res  = await fetch("/history");
    const data = await res.json();
    const panel = document.getElementById("historyPanel");
    if (!data.length) { panel.style.display = "none"; return; }
    panel.style.display = "";

    document.getElementById("historyList").innerHTML = data.map(h => {
      const inp = h.inputs;
      return `
        <div class="history-row">
          <span class="history-num">#${h.id}</span>
          <span class="history-species ${h.prediction}">${h.prediction}</span>
          <span class="history-inputs">PL:${inp.petal_length} PW:${inp.petal_width}</span>
          <span class="history-conf">${h.confidence}%</span>
          <span class="history-time">${h.time}</span>
        </div>`;
    }).join("");
  } catch (_) {}
}

async function clearHistory() {
  await fetch("/history/clear", { method:"POST" });
  document.getElementById("historyPanel").style.display = "none";
}

document.addEventListener("DOMContentLoaded", () => {
  updateSlider("sl", 5.1, 4.0, 8.0);
  updateSlider("sw", 3.5, 2.0, 4.5);
  updateSlider("pl", 1.4, 1.0, 7.0);
  updateSlider("pw", 0.2, 0.1, 2.5);
});
