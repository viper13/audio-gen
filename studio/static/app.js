(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    activeTab: "create",
    songs: [],
    analyses: [],
    selectedAnalysis: null,
    selectedFile: null,
    presets: [],
    tagGroups: [],
    models: null,
    job: null,
    analysisJob: null,
    system: null,
    currentTrack: null, // { id, title, subtitle, audio_url, download_name }
    waveformPeaks: null,
    modalSong: null,
    audioCtx: null,
  };

  // --- Helpers ---
  function toast(msg, kind = "info") {
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = msg;
    $("toastContainer").appendChild(el);
    setTimeout(() => el.remove(), 4000);
  }

  function fmtTime(sec) {
    if (!Number.isFinite(sec) || sec < 0) return "0:00";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  async function api(url, opts = {}) {
    const isFormData = opts.body instanceof FormData;
    const headers = isFormData ? (opts.headers || {}) : { "Content-Type": "application/json", ...(opts.headers || {}) };
    const res = await fetch(url, { ...opts, headers });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || `Request failed (${res.status})`);
    }
    return data;
  }

  // --- Tab Switching ---
  function switchTab(tab) {
    state.activeTab = tab;
    $("tabCreate").classList.toggle("active", tab === "create");
    $("tabAnalyze").classList.toggle("active", tab === "analyze");
    $("createView").classList.toggle("hidden", tab !== "create");
    $("analyzeView").classList.toggle("hidden", tab !== "analyze");
  }

  // --- System & Model UI ---
  function renderSystem(sys) {
    state.system = sys;
    const rt = sys.runtime || {};
    const mem = sys.memory || {};
    const an = sys.analysis || {};
    const isTorch = (sys.backend || rt.backend) === "torch";

    const brandTag = $("brandTag");
    if (brandTag) {
      brandTag.textContent = isTorch
        ? (rt.cuda ? "CUDA · BF16" : "PyTorch · CPU")
        : "MLX · 8-bit";
    }

    $("deviceLabel").textContent = rt.device_name || (isTorch ? "CUDA GPU" : "Apple Silicon");
    if (isTorch && mem.vram_total_gib != null) {
      $("memoryLabel").textContent = `${mem.vram_available_gib ?? "--"} / ${mem.vram_total_gib} GB VRAM`;
      $("memoryPill").title = `GPU VRAM (${mem.available_gib ?? "--"} / ${mem.total_gib ?? "--"} GB System RAM free)`;
    } else {
      $("memoryLabel").textContent = `${mem.available_gib ?? "--"} / ${mem.total_gib ?? "--"} GB free`;
    }

    const gpuOk = Boolean(rt.supported || rt.metal || rt.cuda);
    const dot = $("metalDot");
    dot.className = "dot " + (gpuOk ? (mem.low_memory_warning ? "dot-warn" : "dot-ok") : "dot-danger");
    $("lowMemNotice").classList.toggle("hidden", !mem.low_memory_warning);
    if (isTorch) {
      $("lowMemNotice").innerHTML =
        "<strong>Low available GPU VRAM detected:</strong> Close other GPU-heavy apps before generating to keep headroom for the YuE2-3B CUDA pipeline.";
      const prec = $("precisionSelect");
      if (prec && !prec.dataset.torchConfigured) {
        prec.dataset.torchConfigured = "1";
        prec.innerHTML = `<option value="bf16" selected>BF16 Full Precision (CUDA · AR Offload)</option>`;
      }
    }

    const warn = $("analysisDepWarning");
    warn.classList.toggle("hidden", an.ready !== false);
    if (an.reason) {
      warn.innerHTML = `<strong>Unavailable on this backend:</strong> ${an.reason}`;
    }
    syncAnalyzeButton();
  }

  function renderModels(models) {
    state.models = models;
    const ready = Boolean(models.ready);
    const dl = models.download || {};
    const isDownloading = dl.status === "downloading";
    const expectedGb = models.expected_size_gib ?? 10.6;
    const isTorch = models.backend === "torch";

    if ($("setupCardTitle")) {
      $("setupCardTitle").textContent = isTorch ? "Download YuE2-3B CUDA Weights" : "Download YuE2-3B MLX Weights";
    }
    if ($("setupCardDesc")) {
      $("setupCardDesc").innerHTML = isTorch
        ? `Official PyTorch weights (<code>${models.repo_id || "m-a-p/YuE2-3B + m-a-p/YuE2-Vae"}</code>, ~${expectedGb} GB): BF16 YuE2-3B AR+NAR and 48kHz Oobleck VAE.`
        : `Pre-converted Apple Silicon weights (<code>${models.repo_id || "vanch007/mlx-Yue2-3B"}</code>, ~${expectedGb} GB): 8-bit AR, BF16 acoustic NAR, and 48kHz Oobleck VAE.`;
    }

    const dot = $("modelDot");
    const label = $("modelLabel");
    const topBtn = $("topDownloadBtn");
    const setupCard = $("modelSetupCard");

    if (ready) {
      dot.className = "dot dot-ok";
      label.textContent = `YuE2-3B Ready (${(models.available_precisions || []).join(", ")})`;
      topBtn.classList.add("hidden");
      setupCard.classList.add("hidden");
    } else if (isDownloading) {
      dot.className = "dot dot-warn";
      label.textContent = `Downloading ${dl.percent || 0}%`;
      topBtn.classList.add("hidden");
      setupCard.classList.remove("hidden");
      $("downloadModelBtn").disabled = true;
      $("downloadModelBtn").textContent = "Downloading...";
      $("downloadProgressWrap").classList.remove("hidden");
      $("downloadStatusText").textContent = dl.message || "Downloading weights...";
      $("downloadPctText").textContent = `${dl.percent || 0}%`;
      $("downloadProgressBar").style.width = `${dl.percent || 0}%`;
      $("downloadErrorText").classList.add("hidden");
    } else {
      dot.className = "dot dot-warn";
      label.textContent = "Model weights needed";
      topBtn.classList.remove("hidden");
      setupCard.classList.remove("hidden");
      $("downloadModelBtn").disabled = false;
      $("downloadModelBtn").textContent = `Download Weights (~${expectedGb} GB)`;
      if (dl.status === "failed" && dl.error) {
        $("downloadErrorText").textContent = dl.error;
        $("downloadErrorText").classList.remove("hidden");
      } else {
        $("downloadErrorText").classList.add("hidden");
      }
    }
    syncGenerateButton();
  }

  async function triggerDownload() {
    try {
      const snap = await api("/api/models/download", { method: "POST" });
      renderModels(snap);
      toast("Started downloading YuE2-3B weights from Hugging Face.", "ok");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // --- Presets & Style Chips ---
  function renderPresetsAndTags(data) {
    state.presets = data.presets || [];
    state.tagGroups = data.tag_groups || [];

    const presetsRow = $("presetsRow");
    presetsRow.innerHTML = "";
    state.presets.forEach((p, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preset-btn";
      btn.innerHTML = `<span class="preset-badge">${p.badge}</span><span class="preset-name">${p.name}</span>`;
      btn.addEventListener("click", () => applyPreset(p));
      presetsRow.appendChild(btn);
      if (idx === 0) applyPreset(p, true);
    });

    const tagGroups = $("tagGroups");
    tagGroups.innerHTML = "";
    state.tagGroups.forEach((group) => {
      const wrap = document.createElement("div");
      wrap.className = "tag-group";
      const cat = document.createElement("div");
      cat.className = "tag-cat";
      cat.textContent = group.category;
      const chips = document.createElement("div");
      chips.className = "tag-chips";

      group.tags.forEach((tag) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = tag;
        chip.addEventListener("click", () => toggleStyleTag(tag));
        chips.appendChild(chip);
      });
      wrap.append(cat, chips);
      tagGroups.appendChild(wrap);
    });
    highlightActiveChips();
  }

  function applyPreset(p, silent = false) {
    $("songTitle").value = p.title || "";
    $("styleInput").value = p.style || "";
    $("lyricsInput").value = p.lyrics || "";
    $("cotSelect").value = p.cot || "melody";
    $("stepsSelect").value = String(p.ode_steps || 8);
    $("precisionSelect").value = p.precision || "8bit";
    if (p.max_tokens) {
      const opts = Array.from($("lengthSelect").options).map((o) => Number(o.value));
      const nearest = opts.reduce((a, b) => (Math.abs(b - p.max_tokens) < Math.abs(a - p.max_tokens) ? b : a));
      $("lengthSelect").value = String(nearest);
    }
    $("instrumentalToggle").checked = Boolean(p.instrumental);
    $("abcInput").value = p.abc || "";
    if (p.abc) $("advancedDrawer").open = true;
    highlightActiveChips();
    if (!silent) toast(`Loaded: ${p.name}`, "ok");
  }

  function toggleStyleTag(tag) {
    const input = $("styleInput");
    const parts = input.value
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const idx = parts.findIndex((s) => s.toLowerCase() === tag.toLowerCase());
    if (idx >= 0) parts.splice(idx, 1);
    else parts.push(tag);
    input.value = parts.join(", ");
    highlightActiveChips();
  }

  function highlightActiveChips() {
    const val = $("styleInput").value.toLowerCase();
    document.querySelectorAll("#tagGroups .chip").forEach((chip) => {
      chip.classList.toggle("active", val.includes(chip.textContent.toLowerCase()));
    });
  }

  // --- Generation Job UI ---
  const STEP_ORDER = ["loading", "planning", "semantic", "synthesis", "decoding"];

  function renderJob(job) {
    state.job = job;
    const panel = $("jobPanel");

    if (!job || job.status === "idle") {
      panel.classList.add("hidden");
      syncGenerateButton();
      return;
    }

    panel.classList.remove("hidden");
    $("jobTitle").textContent = job.title || "Generating Track";
    $("jobStageLabel").textContent = job.stage_label || job.status;
    $("jobElapsed").textContent = `${(job.elapsed_seconds ?? 0).toFixed(1)}s`;
    $("jobPct").textContent = `${job.overall_percent ?? 0}%`;
    $("jobProgressBar").style.width = `${job.overall_percent ?? 0}%`;

    const isRunning = job.status === "running";
    $("jobEq").classList.toggle("paused", !isRunning);
    $("cancelJobBtn").classList.toggle("hidden", !isRunning);

    const parts = [];
    if (job.completed) {
      if (job.total) parts.push(`${job.completed}/${job.total} ${job.unit || ""}`.trim());
      else parts.push(`${job.completed} ${job.unit || ""}`.trim());
    }
    if (job.rate > 0 && job.unit === "tokens") parts.push(`${job.rate} tok/s`);
    if (job.estimated_audio_seconds) parts.push(`~${job.estimated_audio_seconds}s audio`);
    $("jobDetailMetrics").textContent = parts.join(" · ") || (isRunning ? "Working on Apple Silicon GPU..." : job.status);

    const activeIdx = job.status === "completed" ? STEP_ORDER.length : STEP_ORDER.indexOf(job.stage);
    document.querySelectorAll("#jobStepper .step").forEach((el) => {
      const idx = STEP_ORDER.indexOf(el.dataset.step);
      el.classList.remove("done", "active");
      if (idx < activeIdx) el.classList.add("done");
      else if (idx === activeIdx && isRunning) el.classList.add("active");
    });

    const errEl = $("jobErrorBanner");
    if (job.status === "failed" && job.error) {
      errEl.textContent = job.error;
      errEl.classList.remove("hidden");
    } else {
      errEl.classList.add("hidden");
    }

    syncGenerateButton();
    syncAnalyzeButton();
  }

  function syncGenerateButton() {
    const btn = $("generateBtn");
    const txt = $("generateBtnText");
    const running = state.job && state.job.status === "running";
    const analyzing = state.analysisJob && state.analysisJob.status === "running";
    const ready = state.models && state.models.ready;

    if (running) {
      btn.disabled = true;
      txt.textContent = "Generating on Apple Silicon...";
    } else if (analyzing) {
      btn.disabled = true;
      txt.textContent = "GPU Busy (Analyzing Audio)...";
    } else if (!ready) {
      btn.disabled = true;
      txt.textContent = "Download YuE2-3B Weights First";
    } else {
      btn.disabled = false;
      txt.textContent = "Generate Song";
    }
  }

  async function submitGenerate(e) {
    if (e) e.preventDefault();
    if ($("generateBtn").disabled) return;

    const payload = {
      title: $("songTitle").value.trim() || "Untitled Track",
      style: $("styleInput").value.trim(),
      lyrics: $("lyricsInput").value,
      cot: $("cotSelect").value,
      precision: $("precisionSelect").value,
      ode_steps: Number($("stepsSelect").value),
      max_tokens: Number($("lengthSelect").value),
      instrumental: $("instrumentalToggle").checked,
      seed: $("seedInput").value.trim() ? Number($("seedInput").value.trim()) : null,
      temperature: Number($("tempInput").value),
      top_p: Number($("topPInput").value),
      repetition_penalty: Number($("repInput").value),
      abc: $("abcInput").value.trim() || null,
    };

    if (!payload.style) {
      toast("Please enter a genre/style prompt.", "error");
      return;
    }
    if (!payload.instrumental && !payload.lyrics.trim()) {
      toast("Please enter lyrics or enable Instrumental mode.", "error");
      return;
    }

    try {
      const job = await api("/api/generate", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      renderJob(job);
      toast(`Started generating "${payload.title}"`, "ok");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // --- Song Library ---
  async function refreshSongs(autoPlaySongId = null) {
    try {
      const data = await api("/api/songs");
      state.songs = data.songs || [];
      renderSongList();
      if (autoPlaySongId) {
        const found = state.songs.find((s) => s.id === autoPlaySongId);
        if (found) playTrackFromSong(found);
      }
    } catch (err) {
      console.error(err);
    }
  }

  function renderSongList() {
    const q = $("searchLibrary").value.trim().toLowerCase();
    const filtered = state.songs.filter((s) => {
      if (!q) return true;
      return (
        (s.title || "").toLowerCase().includes(q) ||
        (s.style || "").toLowerCase().includes(q) ||
        (s.lyrics || "").toLowerCase().includes(q)
      );
    });

    $("libraryCount").textContent = String(state.songs.length);
    const list = $("songList");

    if (!filtered.length) {
      list.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">♫</div>
          <div class="empty-title">${state.songs.length ? "No matching tracks" : "No songs generated yet"}</div>
          <p class="muted">${state.songs.length ? "Try a different filter." : "Pick a preset on the left or write your own lyrics and hit <strong>Generate Song</strong>."}</p>
        </div>`;
      return;
    }

    list.innerHTML = "";
    filtered.forEach((song) => {
      const isCurrent = state.currentTrack && state.currentTrack.id === `song:${song.id}`;
      const isPlaying = isCurrent && !audioEl.paused;

      const card = document.createElement("div");
      card.className = "song-card" + (isCurrent ? " playing" : "");

      const playBtn = document.createElement("button");
      playBtn.type = "button";
      playBtn.className = "play-circle";
      playBtn.textContent = isPlaying ? "❚❚" : "▶";
      playBtn.title = isPlaying ? "Pause" : "Play";
      playBtn.addEventListener("click", () => {
        if (isCurrent) togglePlayPause();
        else playTrackFromSong(song);
      });

      const info = document.createElement("div");
      info.className = "song-info";
      info.innerHTML = `
        <div class="song-title"></div>
        <div class="song-style"></div>
        <div class="badges-row">
          <span class="badge badge-accent mono">${fmtTime(song.audio_seconds)}</span>
          <span class="badge">${song.precision}</span>
          <span class="badge">${song.ode_steps} steps</span>
          <span class="badge">${song.cot}</span>
          ${song.instrumental ? '<span class="badge">Instrumental</span>' : ""}
          ${song.has_abc ? '<span class="badge">ABC Score</span>' : ""}
          <span class="badge mono">${song.file_size_mb} MB</span>
        </div>`;
      info.querySelector(".song-title").textContent = song.title;
      info.querySelector(".song-style").textContent = song.style;

      const actions = document.createElement("div");
      actions.className = "song-actions";

      const inspectBtn = document.createElement("button");
      inspectBtn.type = "button";
      inspectBtn.className = "btn-icon";
      inspectBtn.title = "Inspect Lyrics & ABC Score";
      inspectBtn.textContent = "☰";
      inspectBtn.addEventListener("click", () => openDetailModal(song.id));

      const dlLink = document.createElement("a");
      dlLink.className = "btn-icon";
      dlLink.href = song.audio_url;
      dlLink.download = `${song.id}.flac`;
      dlLink.title = "Download 48kHz FLAC";
      dlLink.textContent = "↓";

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "btn-icon";
      delBtn.title = "Delete track";
      delBtn.textContent = "🗑";
      delBtn.addEventListener("click", () => handleDeleteSong(song));

      actions.append(inspectBtn, dlLink, delBtn);
      card.append(playBtn, info, actions);
      list.appendChild(card);
    });
  }

  async function handleDeleteSong(song) {
    if (!confirm(`Delete "${song.title}" permanently?`)) return;
    try {
      await api(`/api/songs/${encodeURIComponent(song.id)}`, { method: "DELETE" });
      if (state.currentTrack && state.currentTrack.id === `song:${song.id}`) {
        audioEl.pause();
        state.currentTrack = null;
        $("playerBar").classList.add("hidden");
      }
      toast(`Deleted "${song.title}"`, "ok");
      await refreshSongs();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // --- Audio Analysis & Transcription ---
  function setSelectedFile(file) {
    if (!file) return;
    state.selectedFile = file;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    $("dropTitle").textContent = file.name;
    $("dropSub").textContent = `${sizeMb} MB · Ready to analyze locally on Apple Silicon`;
    if (!$("analysisTitle").value.trim()) {
      $("analysisTitle").value = file.name.replace(/\.[^/.]+$/, "");
    }
    syncAnalyzeButton();
  }

  function syncAnalyzeButton() {
    const btn = $("analyzeBtn");
    const txt = $("analyzeBtnText");
    const analyzing = state.analysisJob && state.analysisJob.status === "running";
    const generating = state.job && state.job.status === "running";
    const depsReady = !(state.system && state.system.analysis && state.system.analysis.ready === false);

    if (analyzing) {
      btn.disabled = true;
      txt.textContent = "Analyzing Audio on MLX...";
    } else if (generating) {
      btn.disabled = true;
      txt.textContent = "GPU Busy (Generating Song)...";
    } else if (!depsReady) {
      btn.disabled = true;
      txt.textContent = "Install ffmpeg / transcription extras first";
    } else if (!state.selectedFile) {
      btn.disabled = true;
      txt.textContent = "Select an Audio File to Analyze";
    } else {
      btn.disabled = false;
      txt.textContent = `Analyze "${state.selectedFile.name}"`;
    }
  }

  function renderAnalysisJob(job) {
    state.analysisJob = job;
    const panel = $("analysisJobPanel");
    if (!job || job.status === "idle") {
      panel.classList.add("hidden");
      syncAnalyzeButton();
      syncGenerateButton();
      return;
    }

    panel.classList.remove("hidden");
    const isRunning = job.status === "running";
    $("analysisJobTitle").textContent = job.title || "Analyzing Audio";
    $("analysisJobStage").textContent = job.stage_label || job.status;
    $("analysisJobElapsed").textContent = `${(job.elapsed_seconds ?? 0).toFixed(1)}s`;
    $("analysisJobPct").textContent = `${job.percent ?? 0}%`;
    $("analysisJobBar").style.width = `${job.percent ?? 0}%`;
    $("analysisEq").classList.toggle("paused", !isRunning);
    $("cancelAnalysisBtn").classList.toggle("hidden", !isRunning);

    const bits = [];
    if (job.windows) bits.push(`Window ${job.window}/${job.windows}`);
    if (job.tokens) bits.push(`${job.tokens} events`);
    $("analysisJobMetrics").textContent = bits.join(" · ") || (isRunning ? "SheetSage2 + MERT-v2 MLX" : job.status);

    const errEl = $("analysisJobError");
    if (job.status === "failed" && job.error) {
      errEl.textContent = job.error;
      errEl.classList.remove("hidden");
    } else {
      errEl.classList.add("hidden");
    }

    syncAnalyzeButton();
    syncGenerateButton();
  }

  async function submitAnalyze(e) {
    e.preventDefault();
    if (!state.selectedFile || $("analyzeBtn").disabled) return;
    const fd = new FormData();
    fd.append("file", state.selectedFile);
    fd.append("title", $("analysisTitle").value.trim());
    fd.append("task", $("analysisTask").value);
    fd.append("max_seconds", $("analysisMaxSec").value);

    try {
      const job = await api("/api/analyses/upload", { method: "POST", body: fd });
      renderAnalysisJob(job);
      toast(`Started analyzing "${state.selectedFile.name}"`, "ok");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function refreshAnalyses(selectId = null) {
    try {
      const data = await api("/api/analyses");
      state.analyses = data.analyses || [];
      renderAnalysesList();
      if (selectId) {
        await loadAnalysisDetail(selectId);
      } else if (!state.selectedAnalysis && state.analyses.length > 0) {
        await loadAnalysisDetail(state.analyses[0].id);
      }
    } catch (err) {
      console.error(err);
    }
  }

  function renderAnalysesList() {
    $("analysesCount").textContent = String(state.analyses.length);
    const list = $("analysesList");
    if (!state.analyses.length) {
      list.innerHTML = `<div class="empty-state compact-empty"><p class="muted">No analyzed tracks yet.</p></div>`;
      return;
    }
    list.innerHTML = "";
    state.analyses.forEach((item) => {
      const active = state.selectedAnalysis && state.selectedAnalysis.id === item.id;
      const card = document.createElement("div");
      card.className = "song-card" + (active ? " playing" : "");
      card.style.cursor = "pointer";
      card.addEventListener("click", () => loadAnalysisDetail(item.id));

      const info = document.createElement("div");
      info.className = "song-info";
      info.innerHTML = `
        <div class="song-title"></div>
        <div class="badges-row">
          <span class="badge badge-accent mono">${item.bpm ? item.bpm + " BPM" : "—"}</span>
          <span class="badge mono">${item.key || "—"}</span>
          <span class="badge mono">${fmtTime(item.duration_seconds)}</span>
          <span class="badge">${item.chord_count} chords</span>
        </div>`;
      info.querySelector(".song-title").textContent = item.title;

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "btn-icon";
      delBtn.title = "Delete analysis";
      delBtn.textContent = "🗑";
      delBtn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        if (!confirm(`Delete analysis "${item.title}"?`)) return;
        await api(`/api/analyses/${encodeURIComponent(item.id)}`, { method: "DELETE" });
        if (state.selectedAnalysis && state.selectedAnalysis.id === item.id) {
          state.selectedAnalysis = null;
          $("analysisResultView").classList.add("hidden");
          $("analysisEmptyInspector").classList.remove("hidden");
        }
        await refreshAnalyses();
      });

      card.append(info, delBtn);
      list.appendChild(card);
    });
  }

  const STRUCT_COLORS = {
    intro: "#60a5fa",
    verse: "#34d399",
    chorus: "#f59e0b",
    bridge: "#fb7185",
    outro: "#a78bfa",
    instrumental: "#38bdf8",
    solo: "#f472b6",
  };

  function structColor(label) {
    const low = (label || "").toLowerCase();
    for (const [k, col] of Object.entries(STRUCT_COLORS)) {
      if (low.includes(k)) return col;
    }
    return "#94a3b8";
  }

  async function loadAnalysisDetail(analysisId) {
    try {
      const detail = await api(`/api/analyses/${encodeURIComponent(analysisId)}`);
      state.selectedAnalysis = detail;
      renderAnalysesList();

      $("analysisEmptyInspector").classList.add("hidden");
      $("analysisResultView").classList.remove("hidden");

      $("resTitle").textContent = detail.title;
      $("resSub").textContent = `${detail.source_filename || "Audio"} · analyzed in ${detail.elapsed_seconds}s (${detail.task})`;
      $("resPlayBtn").classList.toggle("hidden", !detail.audio_url);

      $("kpiBpm").textContent = detail.bpm ? `${detail.bpm} BPM` : "—";
      $("kpiKey").textContent = detail.key || "—";
      $("kpiMeter").textContent = detail.meter || "—";
      $("kpiDuration").textContent = fmtTime(detail.duration_seconds);
      $("kpiChords").textContent = String((detail.chords || []).length);
      $("kpiNotes").textContent = String(detail.melody_notes || 0);

      // Structure timeline
      const tl = $("structureTimeline");
      const legend = $("structureLegend");
      tl.innerHTML = "";
      legend.innerHTML = "";
      const structs = detail.structures || [];
      const dur = Math.max(1, detail.duration_seconds || 1);
      if (structs.length) {
        structs.forEach((seg) => {
          const pct = Math.max(3, ((seg.end - seg.start) / dur) * 100);
          const div = document.createElement("div");
          div.className = "struct-seg";
          div.style.width = `${pct}%`;
          div.style.background = structColor(seg.label);
          div.textContent = seg.label;
          div.title = `${seg.label} (${fmtTime(seg.start)} – ${fmtTime(seg.end)})`;
          div.addEventListener("click", () => seekAnalysisAudio(detail, seg.start));
          tl.appendChild(div);

          const tag = document.createElement("span");
          tag.className = "badge mono";
          tag.textContent = `${fmtTime(seg.start)} ${seg.label}`;
          legend.appendChild(tag);
        });
      } else {
        tl.innerHTML = `<div class="muted" style="padding:8px 12px;font-size:12px">No structure segments detected</div>`;
      }

      // Chords grid
      const cg = $("chordsGrid");
      cg.innerHTML = "";
      const chords = (detail.chords || []).filter((c) => c.label && c.label !== "N" && c.label !== "X");
      $("chordCountHint").textContent = `${chords.length} chords`;
      if (chords.length) {
        chords.forEach((c) => {
          const pill = document.createElement("button");
          pill.type = "button";
          pill.className = "chord-pill";
          pill.innerHTML = `<span class="chord-time">${fmtTime(c.start)}</span><span class="chord-name"></span>`;
          pill.querySelector(".chord-name").textContent = c.label;
          pill.addEventListener("click", () => seekAnalysisAudio(detail, c.start));
          cg.appendChild(pill);
        });
      } else {
        cg.innerHTML = `<span class="muted">No chords extracted (use Full Analysis mode for chord recognition).</span>`;
      }

      // Export links
      const ex = $("exportButtons");
      ex.innerHTML = "";
      const files = [
        ...(detail.has_abc ? [["score.abc", "ABC Score (.abc)"]] : []),
        ...(detail.midi_files || []).map((f) => [f, f]),
        ["chord.lab", "Chords (.lab)"],
        ["structure.lab", "Structure (.lab)"],
      ];
      files.forEach(([fname, label]) => {
        const a = document.createElement("a");
        a.className = "btn-secondary";
        a.href = `/api/analyses/${encodeURIComponent(detail.id)}/file/${encodeURIComponent(fname)}`;
        a.download = `${detail.id}-${fname}`;
        a.textContent = `${label} ↓`;
        ex.appendChild(a);
      });

      // ABC score
      const hasAbc = Boolean(detail.abc && detail.abc.trim());
      $("resAbcBlock").classList.toggle("hidden", !hasAbc);
      if (hasAbc) $("resAbcCode").textContent = detail.abc;
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function seekAnalysisAudio(detail, seconds) {
    if (!detail.audio_url) return;
    const trackId = `analysis:${detail.id}`;
    if (!state.currentTrack || state.currentTrack.id !== trackId) {
      playTrack({
        id: trackId,
        title: detail.title,
        subtitle: `Analyzed Audio · ${detail.bpm || "--"} BPM · Key ${detail.key || "--"}`,
        audio_url: detail.audio_url,
        download_name: detail.source_filename || `${detail.id}.audio`,
      }).then(() => {
        audioEl.currentTime = seconds;
      });
    } else {
      audioEl.currentTime = seconds;
      if (audioEl.paused) audioEl.play().catch(() => {});
    }
  }

  function loadAnalysisIntoCoverComposer() {
    const d = state.selectedAnalysis;
    if (!d) return;
    const styleHints = [];
    if (d.key) styleHints.push(`Key of ${d.key}`);
    if (d.bpm) styleHints.push(`${Math.round(d.bpm)} BPM`);
    styleHints.push("modern studio arrangement, expressive vocals");

    const sectionLyrics = (d.structures || [])
      .map((s) => `[${s.label.charAt(0).toUpperCase() + s.label.slice(1)}]`)
      .join("\n\n");

    applyPreset({
      name: `${d.title} (Cover)`,
      title: `${d.title} (YuE2 Cover)`,
      style: styleHints.join(", "),
      lyrics: sectionLyrics || "[Verse]\nWrite new lyrics or toggle Instrumental for an instrumental cover...\n\n[Chorus]\n...",
      cot: d.task === "full" ? "full" : "melody",
      ode_steps: 8,
      precision: "8bit",
      abc: d.abc || "",
    });
    switchTab("create");
    toast("Loaded extracted ABC score & tempo into Composer! Choose a new genre/style and generate.", "ok");
  }

  // --- Detail Modal ---
  async function openDetailModal(songId) {
    try {
      const detail = await api(`/api/songs/${encodeURIComponent(songId)}`);
      state.modalSong = detail;
      $("modalTitle").textContent = detail.title;
      $("modalStyle").textContent = detail.style;
      $("modalLyrics").textContent = detail.lyrics || "(Instrumental)";

      $("modalBadges").innerHTML = `
        <span class="badge badge-accent mono">${fmtTime(detail.audio_seconds)}</span>
        <span class="badge">${detail.precision}</span>
        <span class="badge">${detail.ode_steps} ODE steps</span>
        <span class="badge">mode: ${detail.cot}</span>
        <span class="badge mono">seed: ${detail.seed ?? "—"}</span>
      `;

      const hasAbc = Boolean(detail.abc && detail.abc.trim());
      $("modalAbcSection").classList.toggle("hidden", !hasAbc);
      if (hasAbc) $("modalAbc").textContent = detail.abc;

      const t = detail.timing || {};
      $("modalStats").innerHTML = `
        <div class="stat-item"><span>Duration</span>${detail.audio_seconds}s</div>
        <div class="stat-item"><span>Total Gen Time</span>${detail.e2e_seconds}s</div>
        <div class="stat-item"><span>NAR Synthesis</span>${t.nar_seconds ? t.nar_seconds.toFixed(1) + "s" : "—"}</div>
        <div class="stat-item"><span>48kHz VAE Decode</span>${t.vae_seconds ? t.vae_seconds.toFixed(1) + "s" : "—"}</div>
      `;

      $("modalDownloadLink").href = detail.audio_url;
      $("modalDownloadLink").download = `${detail.id}.flac`;
      $("detailModal").classList.remove("hidden");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // --- Persistent Waveform Audio Player ---
  const audioEl = $("audioEl");

  function playTrackFromSong(song) {
    return playTrack({
      id: `song:${song.id}`,
      title: song.title,
      subtitle: `${song.precision} · ${song.ode_steps} steps · 48kHz FLAC`,
      audio_url: song.audio_url,
      download_name: `${song.id}.flac`,
    });
  }

  async function playTrack(track) {
    state.currentTrack = track;
    state.waveformPeaks = null;
    $("playerBar").classList.remove("hidden");
    $("playerTitle").textContent = track.title;
    $("playerSubtitle").textContent = track.subtitle;
    $("playerDownloadLink").href = track.audio_url;
    $("playerDownloadLink").download = track.download_name;

    audioEl.src = track.audio_url;
    audioEl.volume = Number($("playerVolume").value);
    try {
      await audioEl.play();
    } catch (_) {}
    renderSongList();
    drawWaveform();
    decodeWaveform(track.audio_url);
  }

  function togglePlayPause() {
    if (!state.currentTrack) return;
    if (audioEl.paused) audioEl.play().catch(() => {});
    else audioEl.pause();
  }

  async function decodeWaveform(url) {
    try {
      const resp = await fetch(url);
      const buf = await resp.arrayBuffer();
      if (!state.audioCtx) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        state.audioCtx = new AudioCtx();
      }
      const decoded = await state.audioCtx.decodeAudioData(buf);
      const ch = decoded.getChannelData(0);
      const bars = 120;
      const block = Math.max(1, Math.floor(ch.length / bars));
      const peaks = new Float32Array(bars);
      let maxVal = 0.001;
      for (let i = 0; i < bars; i++) {
        let sum = 0;
        const start = i * block;
        const end = Math.min(ch.length, start + block);
        for (let j = start; j < end; j += 4) {
          const v = Math.abs(ch[j]);
          if (v > sum) sum = v;
        }
        peaks[i] = sum;
        if (sum > maxVal) maxVal = sum;
      }
      for (let i = 0; i < bars; i++) peaks[i] /= maxVal;
      state.waveformPeaks = peaks;
      drawWaveform();
    } catch (_) {
      const bars = 120;
      const peaks = new Float32Array(bars);
      for (let i = 0; i < bars; i++) {
        peaks[i] = 0.25 + 0.65 * Math.abs(Math.sin(i * 0.28) * Math.cos(i * 0.11));
      }
      state.waveformPeaks = peaks;
      drawWaveform();
    }
  }

  function drawWaveform() {
    const canvas = $("waveformCanvas");
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(200, Math.floor(rect.width));
    const height = 44;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const peaks = state.waveformPeaks || Float32Array.from({ length: 100 }, (_, i) => 0.18 + 0.25 * Math.abs(Math.sin(i * 0.3)));
    const progress = audioEl.duration ? audioEl.currentTime / audioEl.duration : 0;
    const barCount = peaks.length;
    const gap = 2;
    const barWidth = Math.max(2, (width - gap * (barCount - 1)) / barCount);

    for (let i = 0; i < barCount; i++) {
      const x = i * (barWidth + gap);
      const h = Math.max(4, peaks[i] * (height - 6));
      const y = (height - h) / 2;
      const played = i / barCount <= progress;
      ctx.fillStyle = played ? "#f59e0b" : "rgba(255, 255, 255, 0.22)";
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, h, 2);
      ctx.fill();
    }
  }

  // --- SSE Real-Time Stream ---
  function connectEvents() {
    const es = new EventSource("/api/events");
    es.addEventListener("system", (e) => renderSystem(JSON.parse(e.data)));
    es.addEventListener("models", (e) => renderModels(JSON.parse(e.data)));
    es.addEventListener("job", (e) => {
      const prevStatus = state.job && state.job.status;
      const next = JSON.parse(e.data);
      renderJob(next);
      if (prevStatus === "running" && next.status === "completed") {
        toast(`"${next.title}" is ready to play!`, "ok");
      } else if (prevStatus === "running" && next.status === "failed") {
        toast(`Generation failed: ${next.error || "Unknown error"}`, "error");
      }
    });
    es.addEventListener("analysis_job", (e) => {
      const prev = state.analysisJob && state.analysisJob.status;
      const next = JSON.parse(e.data);
      renderAnalysisJob(next);
      if (prev === "running" && next.status === "completed") {
        toast(`Analysis complete for "${next.title}"!`, "ok");
      } else if (prev === "running" && next.status === "failed") {
        toast(`Analysis failed: ${next.error || "Unknown error"}`, "error");
      }
    });
    es.addEventListener("songs_updated", (e) => {
      const payload = JSON.parse(e.data);
      refreshSongs(payload.song_id || null);
    });
    es.addEventListener("analyses_updated", (e) => {
      const payload = JSON.parse(e.data);
      refreshAnalyses(payload.analysis_id || null);
    });
    es.onerror = () => {
      es.close();
      setTimeout(connectEvents, 3000);
    };
  }

  // --- Wire Events ---
  $("tabCreate").addEventListener("click", () => switchTab("create"));
  $("tabAnalyze").addEventListener("click", () => switchTab("analyze"));

  $("generateForm").addEventListener("submit", submitGenerate);
  $("downloadModelBtn").addEventListener("click", triggerDownload);
  $("topDownloadBtn").addEventListener("click", triggerDownload);
  $("cancelJobBtn").addEventListener("click", async () => {
    try {
      await api("/api/jobs/cancel", { method: "POST" });
      toast("Cancelling generation...", "info");
    } catch (err) {
      toast(err.message, "error");
    }
  });

  // Analysis Upload & Controls
  const dropZone = $("dropZone");
  const fileInput = $("audioFileInput");
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) setSelectedFile(e.target.files[0]);
  });
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setSelectedFile(e.dataTransfer.files[0]);
  });

  $("analyzeForm").addEventListener("submit", submitAnalyze);
  $("cancelAnalysisBtn").addEventListener("click", async () => {
    try {
      await api("/api/analyses/cancel", { method: "POST" });
      toast("Cancelling analysis...", "info");
    } catch (err) {
      toast(err.message, "error");
    }
  });
  $("resPlayBtn").addEventListener("click", () => {
    const d = state.selectedAnalysis;
    if (!d || !d.audio_url) return;
    const tid = `analysis:${d.id}`;
    if (state.currentTrack && state.currentTrack.id === tid) togglePlayPause();
    else {
      playTrack({
        id: tid,
        title: d.title,
        subtitle: `Analyzed Audio · ${d.bpm || "--"} BPM · Key ${d.key || "--"}`,
        audio_url: d.audio_url,
        download_name: d.source_filename || `${d.id}.audio`,
      });
    }
  });
  $("resCoverBtn").addEventListener("click", loadAnalysisIntoCoverComposer);
  $("copyResAbcBtn").addEventListener("click", async () => {
    if (state.selectedAnalysis && state.selectedAnalysis.abc) {
      await navigator.clipboard.writeText(state.selectedAnalysis.abc);
      toast("Copied extracted ABC score to clipboard", "ok");
    }
  });

  $("styleInput").addEventListener("input", highlightActiveChips);
  $("clearStyleBtn").addEventListener("click", () => {
    $("styleInput").value = "";
    highlightActiveChips();
  });
  $("clearAbcBtn").addEventListener("click", () => {
    $("abcInput").value = "";
    toast("Cleared custom ABC score", "info");
  });

  document.querySelectorAll(".chip-insert").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ta = $("lyricsInput");
      const sec = btn.dataset.section;
      const start = ta.selectionStart ?? ta.value.length;
      const prefix = start > 0 && !ta.value.slice(0, start).endsWith("\n\n") ? "\n\n" : "";
      const insert = `${prefix}${sec}\n`;
      ta.value = ta.value.slice(0, start) + insert + ta.value.slice(ta.selectionEnd ?? start);
      ta.focus();
      ta.selectionStart = ta.selectionEnd = start + insert.length;
    });
  });

  $("randomSeedBtn").addEventListener("click", () => {
    $("seedInput").value = String(Math.floor(Math.random() * 2_000_000_000));
  });
  $("tempInput").addEventListener("input", (e) => ($("tempVal").textContent = e.target.value));
  $("topPInput").addEventListener("input", (e) => ($("topPVal").textContent = e.target.value));
  $("repInput").addEventListener("input", (e) => ($("repVal").textContent = e.target.value));
  $("searchLibrary").addEventListener("input", renderSongList);

  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && state.activeTab === "create") {
      submitGenerate();
    }
  });

  // Audio player events
  $("playerPlayBtn").addEventListener("click", togglePlayPause);
  audioEl.addEventListener("play", () => {
    $("playerPlayBtn").textContent = "❚❚";
    renderSongList();
  });
  audioEl.addEventListener("pause", () => {
    $("playerPlayBtn").textContent = "▶";
    renderSongList();
  });
  audioEl.addEventListener("loadedmetadata", () => {
    $("playerDuration").textContent = fmtTime(audioEl.duration);
    drawWaveform();
  });
  audioEl.addEventListener("timeupdate", () => {
    $("playerCurrentTime").textContent = fmtTime(audioEl.currentTime);
    drawWaveform();
  });
  $("playerVolume").addEventListener("input", (e) => {
    audioEl.volume = Number(e.target.value);
  });
  $("playerLoopBtn").addEventListener("click", () => {
    audioEl.loop = !audioEl.loop;
    $("playerLoopBtn").classList.toggle("active", audioEl.loop);
  });
  $("waveformContainer").addEventListener("click", (e) => {
    if (!audioEl.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    audioEl.currentTime = frac * audioEl.duration;
    drawWaveform();
  });
  window.addEventListener("resize", drawWaveform);

  // Modal events
  $("modalCloseBtn").addEventListener("click", () => $("detailModal").classList.add("hidden"));
  $("detailModal").addEventListener("click", (e) => {
    if (e.target === $("detailModal")) $("detailModal").classList.add("hidden");
  });
  $("copyAbcBtn").addEventListener("click", async () => {
    if (state.modalSong && state.modalSong.abc) {
      await navigator.clipboard.writeText(state.modalSong.abc);
      toast("Copied ABC score to clipboard", "ok");
    }
  });
  $("modalRemixBtn").addEventListener("click", () => {
    if (!state.modalSong) return;
    const s = state.modalSong;
    applyPreset({
      name: s.title,
      title: `${s.title} (Remix)`,
      style: s.style,
      lyrics: s.lyrics,
      cot: s.cot,
      ode_steps: s.ode_steps,
      precision: s.precision,
      instrumental: s.instrumental,
      abc: s.abc || "",
    });
    $("detailModal").classList.add("hidden");
    switchTab("create");
  });

  // --- Bootstrap ---
  async function init() {
    try {
      const [presets, sys, models, job, analysisJob] = await Promise.all([
        api("/api/presets"),
        api("/api/system"),
        api("/api/models"),
        api("/api/jobs/current"),
        api("/api/analyses/job"),
      ]);
      renderPresetsAndTags(presets);
      renderSystem(sys);
      renderModels(models);
      renderJob(job);
      renderAnalysisJob(analysisJob);
      await Promise.all([refreshSongs(), refreshAnalyses()]);
      connectEvents();
    } catch (err) {
      toast(`Startup error: ${err.message}`, "error");
    }
  }

  init();
})();
