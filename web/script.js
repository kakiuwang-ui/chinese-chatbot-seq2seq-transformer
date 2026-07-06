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

  /* ===================================================================
     直观动画演示 —— 进入视口自动播放一次（reduceMotion 下直接跳终态）
     =================================================================== */
  function playOnEnter(el, cb) {
    if (!el) return;
    if (reduceMotion) { cb(true); return; }
    const io = new IntersectionObserver((ents) => {
      ents.forEach(e => {
        if (e.isIntersecting) { cb(false); io.unobserve(e.target); }
      });
    }, { threshold: 0.35 });
    io.observe(el);
  }

  // 可清空的定时器集合，供动画重播时取消上一轮尚未触发的步骤
  function scheduler() {
    const ts = [];
    return {
      after(ms, fn) { ts.push(setTimeout(fn, ms)); },
      every(ms, fn) { const id = setInterval(fn, ms); ts.push(id); return id; },
      clear() { ts.forEach(id => { clearTimeout(id); clearInterval(id); }); ts.length = 0; }
    };
  }

  // 给动画图加一个「↻ 重播」按钮（reduceMotion 下不加——无动画可播）
  function addReplayBtn(fig, onClick) {
    if (!fig || reduceMotion) return;
    fig.style.position = fig.style.position || "relative";
    const b = document.createElement("button");
    b.type = "button"; b.className = "anim-replay"; b.textContent = "↻ 重播";
    b.addEventListener("click", onClick);
    fig.appendChild(b);
  }

  // ①a Encoder/Decoder 按时间展开：hidden 向右流动 + 自回归生成
  function initRnnAnim() {
    const stage = document.getElementById("rnn-stage");
    const now = document.getElementById("rnn-now");
    if (!stage) return;
    const enc = ["你", "今天", "心情", "怎样"];   // 4 步输入
    const dec = ["我", "觉", "得", "还", "行"];    // 5 步输出——刻意不等长，强调两边无逐位置对应

    // 构建一条 lane（enc: 单元在上、词在下；dec: 词在上、单元在下），返回 refs
    const buildLane = (tokens, kind) => {
      const row = document.createElement("div");
      row.className = "rnn-row " + kind;
      const cells = [], toks = [], conns = [];
      tokens.forEach((tk, i) => {
        if (i > 0) {
          const cn = document.createElement("div");
          cn.className = "rnn-hconn";
          cn.innerHTML = "<span class='h'>h,c</span><span>→</span>";
          row.appendChild(cn); conns.push(cn);
        }
        const step = document.createElement("div");
        step.className = "rnn-step " + kind;
        const cell = document.createElement("div");
        cell.className = "rnn-cell"; cell.textContent = "LSTM";
        const tok = document.createElement("div");
        tok.className = "rnn-tok" + (kind === "dec" ? " out" : "");
        tok.textContent = tk;
        if (kind === "enc") { step.append(cell, tok); } else { step.append(tok, cell); }
        row.appendChild(step); cells.push(cell); toks.push(tok);
      });
      return { row, cells, toks, conns };
    };

    const encH = document.createElement("div");
    encH.className = "rnn-lane-h";
    encH.innerHTML = "<b>Encoder</b> · 逐步读入问句，(h, c) 向右传递";
    const E = buildLane(enc, "enc");

    const ctx = document.createElement("div");
    ctx.className = "rnn-ctx";
    ctx.innerHTML = "<span class='arrow'>↓</span><span>context = Encoder 末步的 (h, c) — 整句压缩于此，传给 Decoder</span>";

    const decH = document.createElement("div");
    decH.className = "rnn-lane-h dec";
    decH.innerHTML = "<b>Decoder</b> · 从 &lt;bos&gt; 起逐字生成，上一字喂回下一步";
    const D = buildLane(dec, "dec");

    stage.append(encH, E.row, ctx, decH, D.row);

    const lightEnc = (i) => { E.cells[i].classList.add("active"); E.toks[i].classList.add("lit"); if (i > 0) E.conns[i - 1].classList.add("flow"); };
    const lightDec = (i) => { D.cells[i].classList.add("active"); D.toks[i].classList.add("lit"); if (i > 0) D.conns[i - 1].classList.add("flow"); };
    const timers = scheduler();
    const reset = () => {
      timers.clear();
      [].concat(E.cells, D.cells).forEach(c => c.classList.remove("active", "ctx"));
      [].concat(E.toks, D.toks).forEach(t => t.classList.remove("lit"));
      [].concat(E.conns, D.conns).forEach(c => c.classList.remove("flow"));
      ctx.classList.remove("flow");
      if (now) now.textContent = "—";
    };
    const play = (final) => {
      reset();
      if (final) {
        enc.forEach((_, i) => lightEnc(i));
        dec.forEach((_, i) => lightDec(i));
        E.cells[enc.length - 1].classList.add("ctx");
        ctx.classList.add("flow");
        if (now) now.textContent = "展开完成";
        return;
      }
      const STEP = 720;
      let t = 250;
      if (now) now.textContent = "Encoder 读入…";
      enc.forEach((_, i) => { timers.after(t, () => lightEnc(i)); t += STEP; });
      timers.after(t, () => { E.cells[enc.length - 1].classList.add("ctx"); ctx.classList.add("flow"); if (now) now.textContent = "context = 最后的 (h, c)"; });
      t += STEP;
      timers.after(t - STEP + 60, () => { if (now) now.textContent = "Decoder 生成…"; });
      dec.forEach((_, i) => { timers.after(t, () => lightDec(i)); t += STEP; });
      timers.after(t, () => { if (now) now.textContent = "展开完成"; });
    };
    playOnEnter(document.getElementById("rnn-anim"), play);
    addReplayBtn(document.getElementById("rnn-anim"), () => play(false));
  }

  // ① Seq2Seq 瓶颈：输入逐字流入固定向量 → 逐字解出
  function initBottleneck() {
    const stage = document.getElementById("bn-stage");
    if (!stage) return;
    const inp = ["你", "今", "天", "心", "情", "怎", "么", "样"];
    const out = ["那", "就", "和", "我", "聊", "天"];

    const mkCol = (title, chars, cls) => {
      const col = document.createElement("div");
      col.className = "bn-col";
      const t = document.createElement("div");
      t.className = "bn-col-t"; t.textContent = title;
      col.appendChild(t);
      const chips = chars.map(ch => {
        const c = document.createElement("div");
        c.className = "bn-chip " + cls; c.textContent = ch;
        col.appendChild(c);
        return c;
      });
      return { col, chips };
    };
    const mkArrow = () => {
      const a = document.createElement("div");
      a.className = "bn-arrow"; a.textContent = "→";
      return a;
    };

    const cin = mkCol("问句 · 输入", inp, "");
    const box = document.createElement("div");
    box.className = "bn-box";
    box.innerHTML = "固定<br>context<br>向量";
    const cout = mkCol("答句 · 逐字生成", out, "");
    stage.append(cin.col, mkArrow(), box, mkArrow(), cout.col);

    const OVERFLOW_FROM = 6;   // 第 7、8 个字「装不下」
    const timers = scheduler();
    const reset = () => {
      timers.clear();
      cin.chips.forEach(c => c.classList.remove("lit", "in", "overflow"));
      cout.chips.forEach(c => c.classList.remove("lit", "out"));
      box.classList.remove("pulse");
    };
    const play = (final) => {
      reset();
      if (final) {
        cin.chips.forEach((c, i) => { c.classList.add("lit", "in"); if (i >= OVERFLOW_FROM) c.classList.add("overflow"); });
        cout.chips.forEach(c => c.classList.add("lit", "out"));
        return;
      }
      cin.chips.forEach((c, i) => timers.after(250 + i * 300, () => {
        c.classList.add("lit", "in");
        if (i >= OVERFLOW_FROM) c.classList.add("overflow");
        box.classList.remove("pulse");
        void box.offsetWidth;            // 重启动画
        box.classList.add("pulse");
      }));
      const base = 250 + inp.length * 300 + 350;
      cout.chips.forEach((c, i) => timers.after(base + i * 320, () => c.classList.add("lit", "out")));
    };
    playOnEnter(document.getElementById("bn-anim"), play);
    addReplayBtn(document.getElementById("bn-anim"), () => play(false));
  }

  // ② 注意力权重分配：同一「它/小猫」例子，① 打分 → ② softmax → ③ 混合
  function initAttnEg() {
    const rowsEl = document.getElementById("eg-rows");
    const stage = document.getElementById("eg-stage");
    const query = document.getElementById("eg-query");
    const concl = document.getElementById("eg-concl");
    if (!rowsEl) return;
    const data = [
      { w: "小猫", s: "8.0", wt: 82, key: true },
      { w: "牛奶", s: "3.0", wt: 10 },
      { w: "满足", s: "2.0", wt: 5 },
      { w: "喝", s: "1.5", wt: 3 },
    ];
    const head = document.createElement("div");
    head.className = "eg-row eg-head";
    head.innerHTML = "<span>词（Key / Value）</span><span>Q·Kᵀ 打分</span><span>softmax 权重</span>";
    rowsEl.appendChild(head);
    const rows = data.map(d => {
      const row = document.createElement("div"); row.className = "eg-row";
      const w = document.createElement("span"); w.className = "w" + (d.key ? " key" : ""); w.textContent = d.w;
      const sc = document.createElement("span"); sc.className = "sc"; sc.textContent = d.s;
      const bar = document.createElement("div"); bar.className = "eg-bar";
      const i = document.createElement("i");
      const val = document.createElement("span"); val.textContent = (d.wt / 100).toFixed(2);
      bar.append(i, val); row.append(w, sc, bar); rowsEl.appendChild(row);
      return { sc, i, val, row, wt: d.wt };
    });

    const timers = scheduler();
    const reset = () => {
      timers.clear();
      if (query) query.classList.remove("lit");
      rows.forEach(r => { r.sc.classList.remove("show"); r.i.style.width = "0%"; r.val.classList.remove("show"); r.row.classList.remove("hot"); });
      if (concl) concl.classList.remove("show");
      if (stage) stage.textContent = "—";
    };
    const play = (final) => {
      reset();
      if (query) query.classList.add("lit");
      if (final) {
        rows.forEach(r => { r.sc.classList.add("show"); r.i.style.width = r.wt + "%"; r.val.classList.add("show"); });
        rows[0].row.classList.add("hot");
        if (concl) concl.classList.add("show");
        if (stage) stage.textContent = "指代 → 小猫";
        return;
      }
      if (stage) stage.textContent = "① Q·Kᵀ 逐词打分";
      rows.forEach((r, idx) => timers.after(300 + idx * 380, () => r.sc.classList.add("show")));
      const t2 = 300 + rows.length * 380 + 250;
      timers.after(t2, () => {
        if (stage) stage.textContent = "② softmax 归一成权重";
        rows.forEach(r => { r.i.style.width = r.wt + "%"; r.val.classList.add("show"); });
        rows[0].row.classList.add("hot");
      });
      const t3 = t2 + 1050;
      timers.after(t3, () => { if (stage) stage.textContent = "③ 加权混合 Value"; if (concl) concl.classList.add("show"); });
      timers.after(t3 + 750, () => { if (stage) stage.textContent = "指代 → 小猫"; });
    };
    playOnEnter(document.getElementById("eg-anim"), play);
    addReplayBtn(document.getElementById("eg-anim"), () => play(false));
  }

  // ②c 同一例子的竖条版：把权重铺到整句「小猫喝牛奶，它很满足」上，看「它」看向谁
  function initAttnSent() {
    const barsEl = document.getElementById("sent-bars");
    const toksEl = document.getElementById("sent-toks");
    const stage = document.getElementById("sent-stage");
    if (!barsEl || !toksEl) return;
    const toks = ["小猫", "喝", "牛奶", "，", "它", "很", "满足"];
    const w = [0.82, 0.03, 0.10, 0.00, 0.00, 0.00, 0.05];   // 与上方数值表同口径
    const QIDX = 4;                                          // 「它」是提问者
    const bars = toks.map(() => {
      const wrap = document.createElement("div"); wrap.className = "sent-bar-wrap";
      const b = document.createElement("div"); b.className = "sent-bar";
      wrap.appendChild(b); barsEl.appendChild(wrap); return b;
    });
    const labels = toks.map((t, i) => {
      const s = document.createElement("div"); s.className = "sent-tok" + (i === QIDX ? " q" : "");
      s.textContent = t; toksEl.appendChild(s); return s;
    });

    const grow = () => {
      bars.forEach((b, i) => { b.style.height = Math.max(2, w[i] * 100) + "%"; b.style.backgroundColor = heat(w[i]); });
      labels[0].classList.add("hot");
    };
    const timers = scheduler();
    const reset = () => {
      timers.clear();
      bars.forEach(b => { b.style.height = "0%"; b.style.backgroundColor = ""; });
      labels.forEach(l => l.classList.remove("scan", "hot"));
      if (stage) stage.textContent = "—";
    };
    const play = (final) => {
      reset();
      if (final) { grow(); if (stage) stage.textContent = "它 → 小猫"; return; }
      if (stage) stage.textContent = "① 它 逐词比对 Q·Kᵀ";
      toks.forEach((_, i) => timers.after(200 + i * 240, () => {
        labels.forEach(l => l.classList.remove("scan"));
        labels[i].classList.add("scan");
      }));
      const t2 = 200 + toks.length * 240 + 220;
      timers.after(t2, () => {
        labels.forEach(l => l.classList.remove("scan"));
        if (stage) stage.textContent = "② softmax 权重 → 加权求和";
        grow();
      });
      timers.after(t2 + 950, () => { if (stage) stage.textContent = "它 → 小猫"; });
    };
    playOnEnter(document.getElementById("attn-sent"), play);
    addReplayBtn(document.getElementById("attn-sent"), () => play(false));
  }

  // ③ 三种 mask：全亮 → 挡未来（上三角）→ 挡 <pad> → 定格交集
  function initMaskAnim() {
    const grid = document.getElementById("mask-grid");
    const label = document.getElementById("mask-stage-label");
    if (!grid) return;
    const toks = ["<bos>", "那", "就", "和", "我", "<pad>"];
    const L = toks.length, PAD = L - 1;
    grid.style.gridTemplateColumns = `1.4fr repeat(${L}, 1fr)`;
    const data = [];   // data[i][j]

    for (let r = 0; r <= L; r++) {
      const rowRef = [];
      for (let c = 0; c <= L; c++) {
        const cell = document.createElement("div");
        if (r === 0 || c === 0) {
          cell.className = "mask-cell lbl";
          if (r === 0 && c > 0) { cell.textContent = toks[c - 1]; if (c - 1 === PAD) cell.classList.add("pad"); }
          else if (c === 0 && r > 0) { cell.textContent = toks[r - 1]; if (r - 1 === PAD) cell.classList.add("pad"); }
        } else {
          cell.className = "mask-cell";
          rowRef[c - 1] = cell;
        }
        grid.appendChild(cell);
      }
      if (r > 0) data[r - 1] = rowRef;
    }

    const applyLookAhead = () => { for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) if (j > i) data[i][j].classList.add("blocked"); };
    const applyPad = () => { for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) if ((i === PAD || j === PAD) && !data[i][j].classList.contains("blocked")) data[i][j].classList.add("padblock"); };
    const timers = scheduler();
    const reset = () => {
      timers.clear();
      for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) data[i][j].classList.remove("blocked", "padblock");
    };
    const play = (final) => {
      reset();
      if (final) { applyLookAhead(); applyPad(); if (label) label.textContent = "④ 下三角 ∧ 非 pad"; return; }
      if (label) label.textContent = "① 全部可见";
      timers.after(1000, () => {
        if (label) label.textContent = "② 挡住未来（上三角）";
        for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) if (j > i) timers.after(i * 120, () => data[i][j].classList.add("blocked"));
      });
      timers.after(2600, () => { if (label) label.textContent = "③ 再挡 <pad> 行列"; applyPad(); });
      timers.after(3400, () => { if (label) label.textContent = "④ 下三角 ∧ 非 pad"; });
    };
    playOnEnter(document.getElementById("mask-anim"), play);
    addReplayBtn(document.getElementById("mask-anim"), () => play(false));
  }

  // ④ 位置编码：不同频率的正弦/余弦曲线逐条画出
  function initPEAnim() {
    const svg = document.getElementById("pe-svg");
    if (!svg) return;
    const NS = "http://www.w3.org/2000/svg";
    const X0 = 12, X1 = 428, mid = 75;
    const amp = [46, 40, 34, 28], wl = [380, 200, 118, 66];
    const col = ["#57B98C", "#84D6AB", "#E9B14A", "#E3664A"], ph = [0, Math.PI / 2, 0, Math.PI / 2];
    const paths = amp.map((a, k) => {
      let d = "M";
      for (let x = X0; x <= X1; x += 4) {
        const y = mid - a * Math.sin((x - X0) / wl[k] * 2 * Math.PI + ph[k]);
        d += (x === X0 ? "" : " L") + x.toFixed(1) + " " + y.toFixed(1);
      }
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", d); p.setAttribute("stroke", col[k]);
      svg.appendChild(p);
      p.style.setProperty("--len", p.getTotalLength());
      return p;
    });
    const timers = scheduler();
    const play = (final) => {
      timers.clear();
      // 瞬时归零（关掉过渡再擦除），再逐条画出
      paths.forEach(p => { p.style.transition = "none"; p.classList.remove("drawn"); });
      void svg.clientWidth;                          // 强制重排
      paths.forEach(p => { p.style.transition = ""; });
      if (final) { paths.forEach(p => p.classList.add("drawn")); return; }
      paths.forEach((p, k) => timers.after(k * 280, () => p.classList.add("drawn")));
    };
    playOnEnter(document.getElementById("pe-anim"), play);
    addReplayBtn(document.getElementById("pe-anim"), () => play(false));
  }

  // ⑤ warmup 学习率曲线：沿路径画出（升温→衰减），峰值点最后弹出
  function initWarmupAnim() {
    const warm = document.getElementById("lr-warm");
    const decay = document.getElementById("lr-decay");
    const peak = document.getElementById("lr-peak");
    const fig = document.getElementById("lr-fig");
    if (!warm || !decay) return;
    const lens = [warm, decay].map(p => p.getTotalLength());
    [warm, decay].forEach((p, i) => { p.style.strokeDasharray = lens[i]; });
    if (peak) peak.style.transformOrigin = "132px 36px";
    const timers = scheduler();
    const play = (final) => {
      timers.clear();
      // 归零：关过渡 → 复位 → 重排 → 恢复过渡
      [warm, decay].forEach((p, i) => { p.style.transition = "none"; p.style.strokeDashoffset = lens[i]; });
      if (peak) { peak.style.transition = "none"; peak.style.transform = "scale(0)"; }
      void (warm.getBoundingClientRect && warm.getBoundingClientRect());
      [warm, decay].forEach(p => { p.style.transition = "stroke-dashoffset .9s ease"; });
      if (peak) peak.style.transition = "transform .4s ease";
      if (final) { warm.style.strokeDashoffset = 0; decay.style.strokeDashoffset = 0; if (peak) peak.style.transform = "none"; return; }
      warm.style.strokeDashoffset = 0;
      timers.after(750, () => { decay.style.strokeDashoffset = 0; });
      if (peak) timers.after(1550, () => { peak.style.transform = "none"; });
    };
    playOnEnter(fig, play);
    addReplayBtn(fig, () => play(false));
  }

  document.addEventListener("DOMContentLoaded", () => {
    buildGrid();
    setupReveal();
    setupScrollSpy();
    setupProgress();
    initRnnAnim();
    initBottleneck();
    initAttnEg();
    initAttnSent();
    initMaskAnim();
    initPEAnim();
    initWarmupAnim();
  });
})();
