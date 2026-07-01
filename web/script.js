/* =====================================================================
   签名：注意力矩阵（方格稿纸） + 滚动揭示
   ===================================================================== */
(function () {
  "use strict";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- 注意力热力图示意数据 ----
  const inputs  = ["你", "今", "天", "心", "情", "怎", "么", "样"];
  const outputs = ["那", "就", "和", "我", "聊", "天"];
  // 每个输出字主要对齐到输入的哪一段（中心 + 高斯扩散），生成柔和的对角注意力
  const centers = [4, 4, 5, 0, 6, 1];   // 各输出字关注的输入中心位置
  const weights = outputs.map((_, r) => {
    const row = inputs.map((__, c) => Math.exp(-((c - centers[r]) ** 2) / 3.2));
    const max = Math.max(...row);
    return row.map(v => v / max);        // 归一化到 0..1
  });

  // ---- 多段色阶：纸 → 稿纸绿 → 琥珀 → 朱红 ----
  const stops = [
    [0.00, [22, 22, 22]],     /* 暗格 · 中性 */
    [0.30, [40, 95, 76]],     /* 深青玉 */
    [0.58, [87, 185, 140]],   /* 亮青玉 */
    [0.80, [233, 177, 74]],   /* 琥珀 */
    [1.00, [226, 102, 74]],   /* 朱红 */
  ];
  function heat(w) {
    for (let i = 1; i < stops.length; i++) {
      if (w <= stops[i][0]) {
        const [a0, c0] = stops[i - 1], [a1, c1] = stops[i];
        const t = (w - a0) / (a1 - a0);
        const mix = c0.map((c, k) => Math.round(c + (c1[k] - c) * t));
        return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
      }
    }
    return `rgb(${stops[stops.length - 1][1].join(",")})`;
  }

  function buildGrid() {
    const grid = document.getElementById("attn-grid");
    if (!grid) return;
    const labelW = window.innerWidth < 860 ? "1.6rem" : "2.4rem";
    const cols = `${labelW} repeat(${inputs.length}, minmax(0, 1fr))`;
    const cells = [];

    // 表头：空角 + 输入字
    const head = document.createElement("div");
    head.className = "row";
    head.style.gridTemplateColumns = cols;
    head.appendChild(document.createElement("span"));   // 角
    inputs.forEach(ch => {
      const l = document.createElement("span");
      l.className = "lbl col"; l.textContent = ch;
      head.appendChild(l);
    });
    grid.appendChild(head);

    // 数据行：输出字 + 热力单元
    outputs.forEach((ch, r) => {
      const row = document.createElement("div");
      row.className = "row";
      row.style.gridTemplateColumns = cols;
      const lbl = document.createElement("span");
      lbl.className = "lbl"; lbl.textContent = ch;
      row.appendChild(lbl);
      inputs.forEach((__, c) => {
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.dataset.target = heat(weights[r][c]);
        cells.push(cell);
        row.appendChild(cell);
      });
      grid.appendChild(row);
    });

    // 逐格点亮
    cells.forEach((cell, i) => {
      if (reduceMotion) { cell.style.backgroundColor = cell.dataset.target; return; }
      cell.style.transitionDelay = (i * 22) + "ms";
      requestAnimationFrame(() => requestAnimationFrame(() => {
        cell.style.backgroundColor = cell.dataset.target;
      }));
    });
  }

  // ---- 滚动揭示 ----
  function setupReveal() {
    const els = document.querySelectorAll(".reveal");
    if (reduceMotion || !("IntersectionObserver" in window)) {
      els.forEach(el => el.classList.add("in"));
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { threshold: 0.12 });
    els.forEach(el => io.observe(el));
  }

  // ---- 导航滚动高亮（scrollspy）----
  function setupScrollSpy() {
    const links = Array.from(document.querySelectorAll(".siderail .rail-list a[href^='#']"));
    const map = new Map();
    links.forEach(a => {
      const el = document.getElementById(a.getAttribute("href").slice(1));
      if (el) map.set(el, a);
    });
    if (!map.size || !("IntersectionObserver" in window)) return;

    const spy = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (!e.isIntersecting) return;
        links.forEach(l => l.classList.remove("active"));
        const link = map.get(e.target);
        if (link) link.classList.add("active");
      });
    }, { rootMargin: "-45% 0px -50% 0px", threshold: 0 });
    map.forEach((_, el) => spy.observe(el));
  }

  // ---- 竖栏阅读进度（填充高度）----
  function setupProgress() {
    const bar = document.getElementById("rail-fill");
    if (!bar) return;
    let ticking = false;
    const update = () => {
      const h = document.documentElement.scrollHeight - window.innerHeight;
      const p = h > 0 ? (window.scrollY / h) * 100 : 0;
      bar.style.height = Math.min(100, Math.max(0, p)) + "%";
      ticking = false;
    };
    window.addEventListener("scroll", () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true; }
    }, { passive: true });
    update();
  }

  document.addEventListener("DOMContentLoaded", () => {
    buildGrid();
    setupReveal();
    setupScrollSpy();
    setupProgress();
  });
})();
