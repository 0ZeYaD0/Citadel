/**
 * Citadel UI Controller  (static/js/ui_controller.js)
 *
 * Key improvements over previous version:
 *  - State is hydrated from the backend on load (documents, sessions, notes)
 *  - Messages are persisted to the backend after every send
 *  - Upload uses job polling with multi-stage progress indicators
 *  - Notes are saved / deleted against /api/notes
 *  - Study-tools panel with 5 dedicated actions
 *  - Delete-all button in the sidebar
 *  - Health status badge in the header
 *  - Duplicate upload prevention
 */

// ─── Application state ──────────────────────────────────────────────────────

const state = {
  sources: [],          // { id, name, size, active, status, chunk_count, content }
  sessions: [],         // { id, title, emoji, created_at, updated_at }
  messages: {},         // { [sessionId]: [{ role, text }] }
  currentSessionId: null,
  pinnedNotes: [],      // [{ id, text, session_id, document_id, page_number, tags }]
  uploadInProgress: false,
  queryInProgress: false,

  featuredNotebooks: [
    {
      id: "feat-1",
      author: "Computer Science Dept",
      avatar: "💻",
      title: "Algorithms & Data Structures Master Guide",
      date: "Oct 2026",
      sourceCount: "14 sources",
      gradient: "linear-gradient(135deg, #181A2F 0%, #37415C 50%, #FDA481 100%)",
    },
    {
      id: "feat-2",
      author: "HCI Research Lab",
      avatar: "🎨",
      title: "Human-Centered Interface & Cognitive Design",
      date: "Oct 2026",
      sourceCount: "9 sources",
      gradient: "linear-gradient(135deg, #54162B 0%, #B4182D 60%, #FDA481 100%)",
    },
    {
      id: "feat-3",
      author: "AI Systems Group",
      avatar: "🧠",
      title: "RAG Architectures & Vector Embeddings",
      date: "Oct 2026",
      sourceCount: "11 sources",
      gradient: "linear-gradient(135deg, #0049B7 0%, #242E49 65%, #00DDFF 100%)",
    },
    {
      id: "feat-4",
      author: "Software Engineering",
      avatar: "🚀",
      title: "Build Products People Actually Want",
      date: "Oct 2026",
      sourceCount: "7 sources",
      gradient: "linear-gradient(135deg, #108548 0%, #14241B 95%)",
    },
  ],

  waveColors: {
    "matcha":          ["#108548", "#34D399"],
    "coral-lemonade":  ["#FF1D58", "#F75990"],
    "electric-lagoon": ["#0049B7", "#00DDFF"],
    "sunset-horizon":  ["#FDA481", "#B4182D"],
    "crimson-dusk":    ["#FDA481", "#B4182D"],
    "velvet-ember":    ["#FDA481", "#B4182D"],
  },
};

// ─── DOM references ──────────────────────────────────────────────────────────

const homeHubView         = document.getElementById("homeHubView");
const workspaceView       = document.getElementById("workspaceView");
const featuredGrid        = document.getElementById("featuredGrid");
const recentGrid          = document.getElementById("recentGrid");
const hubSearchInput      = document.getElementById("hubSearchInput");
const sourceListEl        = document.getElementById("sourceList");
const historyListEl       = document.getElementById("historyList");
const chatFeedEl          = document.getElementById("chatFeed");
const promptInputEl       = document.getElementById("promptInput");
const sendBtnEl           = document.getElementById("sendBtn");
const dropzoneEl          = document.getElementById("dropzone");
const fileInputEl         = document.getElementById("fileInput");
const quickAttachBtn      = document.getElementById("quickAttachBtn");
const promptSourcesBar    = document.getElementById("promptSourcesBar");
const activeContextCounter= document.getElementById("activeContextCounter");
const currentSessionTitle = document.getElementById("currentSessionTitle");
const inspectorDrawer     = document.getElementById("inspectorDrawer");
const inspectorContent    = document.getElementById("inspectorContent");
const notesListEl         = document.getElementById("notesList");
const noteCountBadge      = document.getElementById("noteCountBadge");
const themeNameLabel      = document.getElementById("themeNameLabel");
const ambientWaveStage    = document.getElementById("ambientWaveStage");
const waveCanvas          = document.getElementById("waveCanvas");
const uploadProgressOverlay = document.getElementById("uploadProgressOverlay");
const uploadProgressTitle   = document.getElementById("uploadProgressTitle");
const uploadProgressMessage = document.getElementById("uploadProgressMessage");
const uploadProgressDetail  = document.getElementById("uploadProgressDetail");
const healthBadge           = document.getElementById("healthBadge");

// ─── Initialisation ──────────────────────────────────────────────────────────

async function initStudyWorkspace() {
  renderHub();
  setupEvents();
  _syncResponsivePanels();
  window.addEventListener("resize", _syncResponsivePanels);
  initSilkWaveAnimation();
  await hydrateStateFromBackend();
}

/** Load documents, sessions, and notes from the backend on page load. */
async function hydrateStateFromBackend() {
  try {
    // Check health
    const health = await CitadelAPI.healthCheck();
    updateHealthBadge(health);

    // Documents are loaded only after a notebook is opened, so sources remain
    // scoped to the active notebook rather than leaking across notebooks.
    state.sources = [];

    // Restore sessions
    const sessions = await CitadelAPI.getSessions();
    state.sessions = sessions.map((s) => ({
      id: s.id,
      title: s.title,
      emoji: s.emoji || "📓",
      date: _formatDate(s.created_at),
      messages: [],
    }));

    // Restore notes
    const notes = await CitadelAPI.getNotes();
    state.pinnedNotes = notes.map((n) => ({
      id: n.id,
      text: n.text,
      session_id: n.session_id,
      document_id: n.document_id,
      page_number: n.page_number,
      tags: n.tags || [],
    }));

    renderHub();
    renderSources();
    renderNotes();
  } catch (err) {
    console.warn("Could not hydrate state from backend:", err.message);
  }
}

function updateHealthBadge(health) {
  if (!healthBadge) return;
  const ok = health.status === "ok";
  healthBadge.textContent = ok
    ? `● Ready  •  ${health.total_chunks ?? "?"} chunks`
    : "● Not ready";
  healthBadge.className = `health-badge ${ok ? "health-ok" : "health-warn"}`;
  healthBadge.title = ok
    ? `Embeddings: ${health.embedding_model}  |  LLM: ${health.generation_model}`
    : "Pipeline is not initialized. Check your .env and restart the server.";
}

// ─── View navigation ─────────────────────────────────────────────────────────

async function openWorkspaceWithSession(sessionId) {
  state.currentSessionId = sessionId;
  window.CitadelCurrentSessionId = sessionId;
  const session = state.sessions.find((s) => s.id === sessionId);
  if (session && currentSessionTitle) {
    currentSessionTitle.textContent = session.title;
  }
  homeHubView?.classList.add("hidden");
  workspaceView?.classList.remove("hidden");
  showAmbientWave();
  const sessionDocuments = await CitadelAPI.getDocuments(sessionId);
  state.sources = sessionDocuments.map((d) => ({
    id: d.id,
    name: d.filename || d.name,
    size: d.size || _fmtSize(d.file_size_bytes || 0),
    active: d.status === "ready",
    status: d.status,
    chunk_count: d.chunk_count || 0,
    content: d.preview_text || d.content || "",
    file_type: d.file_type,
  }));
  renderSources();
  renderHistory();

  // Load messages from backend for this session
  const msgs = await CitadelAPI.getMessages(sessionId);
  if (!state.messages[sessionId]) state.messages[sessionId] = [];
  if (msgs.length > 0) {
    state.messages[sessionId] = msgs.map((m) => ({ role: m.role, text: m.text }));
    if (session) session.messages = state.messages[sessionId];
  }
  renderChat();
}

async function createNewNotebookAndOpen(customTitle = "Untitled Notebook", emoji = "📓") {
  const today = new Date().toLocaleDateString("en-GB", {
    day: "numeric", month: "short", year: "numeric",
  });

  // Create on backend, get real session ID
  let sid;
  try {
    const res = await CitadelAPI.upsertSession(null, customTitle, emoji);
    sid = res.session_id;
  } catch (err) {
    console.error("Could not create notebook:", err);
    throw err;
  }

  const newSession = { id: sid, title: customTitle, emoji, date: today, messages: [] };
  state.sessions.unshift(newSession);
  state.messages[sid] = [];
  await openWorkspaceWithSession(sid);
}

async function renameCurrentNotebook() {
  if (!state.currentSessionId) return;
  const session = state.sessions.find((item) => item.id === state.currentSessionId);
  if (!session) return;

  beginSessionRename(session, currentSessionTitle);
}

function returnToHub() {
  workspaceView?.classList.add("hidden");
  homeHubView?.classList.remove("hidden");
  renderHub(hubSearchInput ? hubSearchInput.value : "");
}

// ─── Render: Hub ─────────────────────────────────────────────────────────────

function renderHub(searchFilter = "") {
  const query = searchFilter.trim().toLowerCase();

  if (featuredGrid) {
    const filtered = state.featuredNotebooks.filter(
      (f) => f.title.toLowerCase().includes(query) || f.author.toLowerCase().includes(query)
    );
    featuredGrid.innerHTML = filtered
      .map(
        (item) => `
      <div class="featured-card" data-title="${item.title}" style="background: ${item.gradient};">
        <div class="featured-card-top">
          <span class="featured-avatar">${item.avatar}</span>
          <span>${item.author}</span>
        </div>
        <div class="featured-card-bottom">
          <h3>${item.title}</h3>
          <div class="featured-meta">
            <span>${item.date} • ${item.sourceCount}</span>
            <span>🌐</span>
          </div>
        </div>
      </div>`
      )
      .join("");

    featuredGrid.querySelectorAll(".featured-card").forEach((card) => {
      card.addEventListener("click", () => createNewNotebookAndOpen(card.dataset.title, "✨"));
    });
  }

  if (recentGrid) {
    const filteredSessions = state.sessions.filter((s) =>
      s.title.toLowerCase().includes(query)
    );

    const newTileHTML = `
      <div class="recent-card new-card-tile" id="newNotebookTile">
        <div class="new-tile-plus">+</div>
        <strong>Create new notebook</strong>
      </div>`;

    const sessionCardsHTML = filteredSessions
      .map(
        (sess) => `
      <div class="recent-card" data-id="${sess.id}">
        <div class="recent-card-top">
          <span class="recent-emoji">${sess.emoji || "📓"}</span>
          <button class="delete-notebook-btn" data-del="${sess.id}" title="Delete notebook">✕</button>
        </div>
        <div class="recent-card-bottom">
          <h4 class="session-name" title="Click to rename">${sess.title}</h4>
          <span class="recent-card-meta">${sess.date} • ${state.sources.length} source${state.sources.length === 1 ? "" : "s"}</span>
        </div>
      </div>`
      )
      .join("");

    recentGrid.innerHTML = newTileHTML + sessionCardsHTML;

    document.getElementById("newNotebookTile")?.addEventListener("click", () =>
      createNewNotebookAndOpen()
    );

    recentGrid.querySelectorAll(".recent-card[data-id]").forEach((card) => {
      card.addEventListener("click", (e) => {
        if (e.target.classList.contains("delete-notebook-btn")) return;
        openWorkspaceWithSession(card.dataset.id);
      });
    });
    recentGrid.querySelectorAll(".session-name").forEach((name) => {
      name.addEventListener("click", (event) => {
        event.stopPropagation();
        const session = state.sessions.find(
          (item) => item.id === name.closest(".recent-card")?.dataset.id
        );
        if (session) beginSessionRename(session, name);
      });
    });

    recentGrid.querySelectorAll(".delete-notebook-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const delId = btn.dataset.del;
        try { await CitadelAPI.deleteSession(delId); } catch { /* best effort */ }
        state.sessions = state.sessions.filter((s) => s.id !== delId);
        delete state.messages[delId];
        renderHub(hubSearchInput ? hubSearchInput.value : "");
        renderHistory();
      });
    });
  }

}

function beginSessionRename(session, target) {
  if (!target || target.parentElement?.querySelector(".session-name-input")) return;

  const input = document.createElement("input");
  input.className = "session-name-input";
  input.type = "text";
  input.value = session.title;
  input.maxLength = 120;
  if (target.id === "currentSessionTitle") {
    target.hidden = true;
    target.parentElement.appendChild(input);
  } else {
    target.replaceWith(input);
  }
  input.focus();
  input.select();

  let finished = false;
  const finish = async (save) => {
    if (finished) return;
    finished = true;
    const nextTitle = input.value.trim();
    if (save && nextTitle && nextTitle !== session.title) {
      try {
        await CitadelAPI.upsertSession(session.id, nextTitle, session.emoji);
        session.title = nextTitle;
      } catch (error) {
        console.error("Could not rename notebook:", error);
      }
    }
    if (target.id === "currentSessionTitle") {
      target.textContent = session.title;
      target.hidden = false;
      input.remove();
    }
    renderHub();
    renderHistory();
    if (session.id === state.currentSessionId && currentSessionTitle) {
      currentSessionTitle.textContent = session.title;
    }
  };

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") finish(true);
    if (event.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

// ─── Render: Silk wave ───────────────────────────────────────────────────────

function initSilkWaveAnimation() {
  if (!waveCanvas) return;
  const ctx = waveCanvas.getContext("2d");
  let tick = 0;

  function resize() {
    const rect = waveCanvas.parentElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = window.devicePixelRatio || 1;
    waveCanvas.width = rect.width * dpr;
    waveCanvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", resize);

  function drawSilkRibbons() {
    if (!workspaceView || !workspaceView.classList.contains("hidden")) {
      const w = waveCanvas.clientWidth;
      const h = waveCanvas.clientHeight;
      if (w && h && waveCanvas.width !== w * (window.devicePixelRatio || 1)) resize();
      ctx.clearRect(0, 0, w, h);

      const activeTheme = document.documentElement.getAttribute("data-theme") || "sunset-horizon";
      const [colorA, colorB] = state.waveColors[activeTheme] || state.waveColors["sunset-horizon"];
      const totalLines = 34;
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, colorA);
      grad.addColorStop(0.5, colorB);
      grad.addColorStop(1, colorA);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.15;

      for (let i = 0; i < totalLines; i++) {
        const ratio = i / totalLines;
        ctx.globalAlpha = 0.08 + Math.sin(ratio * Math.PI) * 0.26;
        ctx.beginPath();
        for (let x = 0; x <= w; x += 14) {
          const nx = x / w;
          const wave1 = Math.sin(nx * 4.5 + tick + i * 0.09) * 58;
          const wave2 = Math.cos(nx * 2.8 - tick * 0.7 + i * 0.06) * 42;
          const spread = (i - totalLines / 2) * 3.2;
          const y = h * 0.48 + wave1 + wave2 + spread;
          if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      tick += 0.014;
    }
    requestAnimationFrame(drawSilkRibbons);
  }
  drawSilkRibbons();
}

function hideAmbientWave() {
  ambientWaveStage?.classList.add("vanished");
}
function showAmbientWave() {
  ambientWaveStage?.classList.remove("vanished");
}
function getTimeOfDayGreeting() {
  const h = new Date().getHours();
  if (h < 12) return "Good morning.";
  if (h < 18) return "Good afternoon.";
  return "Good evening.";
}

// ─── Render: Sources ─────────────────────────────────────────────────────────

function renderSources() {
  if (!sourceListEl) return;
  sourceListEl.innerHTML = "";
  const activeDocs = state.sources.filter((s) => s.active);

  if (activeContextCounter) {
    activeContextCounter.textContent =
      activeDocs.length === 0
        ? "No documents selected"
        : `${activeDocs.length} document${activeDocs.length === 1 ? "" : "s"} on desk`;
  }

  if (promptSourcesBar) {
    promptSourcesBar.innerHTML = activeDocs.length
      ? activeDocs.map((s) => `<span class="doc-pill">📄 ${s.name}</span>`).join("")
      : `<span class="doc-pill">No files on desk yet</span>`;
  }

  if (state.sources.length === 0) {
    sourceListEl.innerHTML = `<div class="empty-shelf-note">No documents uploaded yet.</div>`;
    return;
  }

  // "Delete all" button when multiple docs exist
  if (state.sources.length > 1) {
    const delAllBtn = document.createElement("button");
    delAllBtn.className = "delete-all-btn";
    delAllBtn.textContent = "Remove all documents";
    delAllBtn.addEventListener("click", async () => {
      if (!confirm("Remove ALL documents? This cannot be undone.")) return;
      try {
        await CitadelAPI.deleteAllDocuments();
        state.sources = [];
        renderSources();
      } catch (err) {
        alert(`Could not delete all documents: ${err.message}`);
      }
    });
    sourceListEl.appendChild(delAllBtn);
  }

  state.sources.forEach((doc, index) => {
    const card = document.createElement("div");
    card.className = `doc-card ${doc.status === "processing" ? "doc-processing" : ""}`;

    const statusIcon =
      doc.status === "processing"
        ? `<span class="doc-status-spinner" title="Processing…"></span>`
        : doc.status === "error"
        ? `<span class="doc-status-error" title="${doc.error || "Error"}">⚠</span>`
        : "";

    card.innerHTML = `
      <div class="doc-card-left">
        <input type="checkbox" class="doc-checkbox" ${doc.active ? "checked" : ""} ${doc.status !== "ready" ? "disabled" : ""} />
        <span class="doc-title" title="${doc.name}">${doc.name}</span>
        ${statusIcon}
      </div>
      <div class="doc-card-actions">
        <span class="doc-size">${doc.size || ""}</span>
        ${doc.chunk_count ? `<span class="doc-chunks" title="Chunks in vector store">${doc.chunk_count} chunks</span>` : ""}
        ${doc.id ? `<button class="doc-delete-btn" type="button" title="Remove document">×</button>` : ""}
      </div>`;

    const checkbox = card.querySelector(".doc-checkbox");
    checkbox?.addEventListener("change", (e) => {
      e.stopPropagation();
      doc.active = e.target.checked;
      renderSources();
    });

    const deleteButton = card.querySelector(".doc-delete-btn");
    deleteButton?.addEventListener("click", async (e) => {
      e.stopPropagation();
      try {
        if (doc.id) await CitadelAPI.deleteDocument(doc.id);
        state.sources.splice(index, 1);
        renderSources();
      } catch (error) {
        console.warn("Could not delete document:", error.message);
      }
    });

    card.addEventListener("click", (e) => {
      if (e.target.tagName !== "INPUT" && !e.target.classList.contains("doc-delete-btn")) {
        openPassageReader(index);
      }
    });

    sourceListEl.appendChild(card);
  });
}

// ─── Render: History ─────────────────────────────────────────────────────────

function renderHistory() {
  if (!historyListEl) return;
  historyListEl.innerHTML = "";

  if (state.sessions.length === 0) {
    historyListEl.innerHTML = `<div class="empty-shelf-note">No past pages yet.</div>`;
    return;
  }

  state.sessions.forEach((sess) => {
    const item = document.createElement("div");
    item.className = `page-item ${sess.id === state.currentSessionId ? "active" : ""}`;
    const title = document.createElement("span");
    title.className = "session-name";
    title.title = "Click to rename";
    title.textContent = sess.title;
    item.appendChild(title);
    title.addEventListener("click", (event) => {
      event.stopPropagation();
      beginSessionRename(sess, title);
    });
    item.addEventListener("click", async () => {
      state.currentSessionId = sess.id;
      if (currentSessionTitle) currentSessionTitle.textContent = sess.title;
      renderHistory();

      // Load messages from backend
      const msgs = await CitadelAPI.getMessages(sess.id);
      state.messages[sess.id] = msgs.map((m) => ({ role: m.role, text: m.text }));
      sess.messages = state.messages[sess.id];
      renderChat();
    });
    historyListEl.appendChild(item);
  });
}

// ─── Render: Chat ─────────────────────────────────────────────────────────────

function renderChat() {
  if (!chatFeedEl) return;
  const session = state.sessions.find((s) => s.id === state.currentSessionId);
  if (!session) return;

  chatFeedEl.innerHTML = `
    <div class="welcome-banner">
      <h2>${getTimeOfDayGreeting()} Ready to think through your notes?</h2>
      <p>Upload your reading material on the left or write a question below.</p>
    </div>`;

  const msgs = state.messages[state.currentSessionId] || [];
  msgs.forEach((msg) => {
    const card = document.createElement("div");
    card.className = `entry-card ${msg.role}`;

    if (msg.role === "user") {
      card.textContent = msg.text;
    } else {
      const evidenceBadge =
        msg.evidence_found === false
          ? `<span class="evidence-badge no-evidence" title="No supporting evidence found in documents">No evidence found</span>`
          : msg.evidence_found === true
          ? `<span class="evidence-badge has-evidence" title="Answer grounded in your documents">Grounded</span>`
          : "";

      card.innerHTML = `
        <div class="entry-meta">
          <span class="entry-author">Citadel Companion</span>
          ${evidenceBadge}
          ${msg.latency_ms ? `<span class="latency-badge" title="Generation latency">${Math.round(msg.latency_ms)}ms</span>` : ""}
          <button class="bookmark-btn" title="Save this thought">Save note</button>
        </div>
        <div class="entry-body">${formatFriendlyText(msg.text)}</div>`;

      const saveBtn = card.querySelector(".bookmark-btn");
      saveBtn.addEventListener("click", async () => {
        const noteText = msg.text.trim();
        try {
          const res = await CitadelAPI.saveNote(noteText, state.currentSessionId);
          state.pinnedNotes.unshift({ id: res.note_id, text: noteText, session_id: state.currentSessionId });
          renderNotes();
          saveBtn.textContent = "✓ Saved";
          saveBtn.disabled = true;
        } catch {
          saveBtn.textContent = "Save note"; // retry allowed
        }
      });

    }

    chatFeedEl.appendChild(card);
  });

  chatFeedEl.scrollTop = chatFeedEl.scrollHeight;
}

// ─── Render: Notes ───────────────────────────────────────────────────────────

function renderNotes() {
  if (noteCountBadge) noteCountBadge.textContent = state.pinnedNotes.length;
  if (!notesListEl) return;

  if (state.pinnedNotes.length === 0) {
    notesListEl.innerHTML = `<div class="reader-empty"><p>No saved highlights yet. Click "Save note" on any response to pin it here.</p></div>`;
    return;
  }
  notesListEl.innerHTML = state.pinnedNotes
    .map(
      (n) => `
    <div class="highlight-card">
      <div class="highlight-text">${n.text}</div>
      <div class="highlight-meta">
        ${n.tags && n.tags.length ? n.tags.map((t) => `<span class="tag-pill">${t}</span>`).join("") : ""}
        <button class="note-delete-btn" data-id="${n.id}" title="Delete note">×</button>
      </div>
    </div>`
    )
    .join("");

  notesListEl.querySelectorAll(".note-delete-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const nid = parseInt(btn.dataset.id, 10);
      try { await CitadelAPI.deleteNote(nid); } catch { /* best effort */ }
      state.pinnedNotes = state.pinnedNotes.filter((n) => n.id !== nid);
      renderNotes();
    });
  });
}

// ─── Format text ─────────────────────────────────────────────────────────────

function formatFriendlyText(text) {
  return (text || "")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.*?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

// ─── Passage reader ───────────────────────────────────────────────────────────

function openPassageReader(docIndex) {
  const doc = state.sources[docIndex];
  if (!doc || !inspectorDrawer || !inspectorContent) return;
  inspectorDrawer.classList.remove("closed");
  workspaceView?.classList.remove("drawer-closed");
  _activateReaderTab("inspector");

  const statusBadge =
    doc.status === "processing"
      ? `<span class="status-pill processing">Processing…</span>`
      : doc.status === "error"
      ? `<span class="status-pill error">Error</span>`
      : `<span class="status-pill ready">${doc.chunk_count || 0} chunks</span>`;

  inspectorContent.innerHTML = `
    <span class="source-badge-meta">SOURCE #${docIndex + 1} • ${doc.size || ""} ${statusBadge}</span>
    <h3 class="reader-doc-heading">${doc.name}</h3>
    <div class="reader-doc-sheet">${doc.content || "(No preview available)"}</div>`;
}

function _activateReaderTab(target) {
  document.querySelectorAll(".reader-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === target);
  });
  document.getElementById("studioTab")?.classList.toggle("hidden", target !== "studio");
  document.getElementById("inspectorTab")?.classList.toggle("hidden", target !== "inspector");
  document.getElementById("notesTab")?.classList.toggle("hidden", target !== "notes");
}

// ─── Send message ─────────────────────────────────────────────────────────────

async function sendUserMessage() {
  const text = promptInputEl.value.trim();
  if (!text) return;
  if (state.queryInProgress) return;

  hideAmbientWave();

  // Ensure we have a session
  if (!state.currentSessionId) {
    await createNewNotebookAndOpen();
  }
  let session = state.sessions.find((s) => s.id === state.currentSessionId);
  if (!session) {
    await createNewNotebookAndOpen();
    session = state.sessions[0];
  }

  if (!state.messages[state.currentSessionId]) {
    state.messages[state.currentSessionId] = [];
  }

  // Auto-name session after first question
  if (
    state.messages[state.currentSessionId].length === 0 &&
    session.title === "Untitled Notebook"
  ) {
    const newTitle = text.slice(0, 36) + (text.length > 36 ? "…" : "");
    session.title = newTitle;
    if (currentSessionTitle) currentSessionTitle.textContent = newTitle;
    try {
      await CitadelAPI.upsertSession(session.id, newTitle, session.emoji);
    } catch { /* non-blocking */ }
    renderHistory();
  }

  state.messages[state.currentSessionId].push({ role: "user", text });
  session.messages = state.messages[state.currentSessionId];

  // Persist user message
  try { await CitadelAPI.addMessage(session.id, "user", text); } catch { /* best effort */ }

  promptInputEl.value = "";
  promptInputEl.style.height = "auto";

  // Show typing indicator
  state.queryInProgress = true;
  sendBtnEl.disabled = true;
  sendBtnEl.innerHTML = `<span class="typing-dots"><span></span><span></span><span></span></span>`;
  renderChat();

  const activeDocs = state.sources.filter((s) => s.active);
  let aiMsg;
  try {
    const res = await CitadelAPI.queryKnowledgeBase(text, activeDocs);
    aiMsg = {
      role: "ai",
      text: res?.answer || "No response received from the model.",
      evidence_found: res?.evidence_found,
      latency_ms: res?.latency_ms,
      citations: res?.citations || [],
    };
  } catch (err) {
    aiMsg = { role: "ai", text: `⚠️ Error: ${err.message}` };
  }

  state.messages[state.currentSessionId].push(aiMsg);
  session.messages = state.messages[state.currentSessionId];

  // Persist AI message
  try { await CitadelAPI.addMessage(session.id, "ai", aiMsg.text); } catch { /* best effort */ }

  state.queryInProgress = false;
  sendBtnEl.disabled = false;
  sendBtnEl.innerHTML = `<span>Ask</span><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>`;
  renderChat();
}

// ─── File upload with job polling ─────────────────────────────────────────────

const _STAGE_LABELS = {
  queued:    { title: "Queuing…",           msg: "Your files are in the queue." },
  extracting:{ title: "Extracting text",    msg: "Reading document content…" },
  chunking:  { title: "Splitting passages", msg: "Breaking text into searchable chunks…" },
  embedding: { title: "Creating embeddings",msg: "Building local vector embeddings…" },
  complete:  { title: "Ready!",             msg: "Your documents are on your desk." },
};

async function handleFileUploads(files) {
  if (!files || !files.length) return;
  if (state.uploadInProgress) return;

  state.uploadInProgress = true;
  const fileCount = files.length;

  _setProgress(
    true,
    fileCount === 1 ? "Adding your resource" : `Adding ${fileCount} resources`,
    "Uploading files…",
    `${fileCount} file${fileCount === 1 ? "" : "s"} selected`
  );

  try {
    // Step 1: upload and get job_id
    const uploadRes = await CitadelAPI.uploadDocuments(files);
    const { job_id, rejected = [] } = uploadRes;

    // Show rejected files
    if (rejected.length) {
      rejected.forEach((r) => console.warn(`Rejected: ${r.name} — ${r.error}`));
    }

    if (!job_id) {
      _setProgress(false);
      state.uploadInProgress = false;
      return;
    }

    // Step 2: poll the job
    await CitadelAPI.pollJob(
      job_id,
      (job) => {
        const label = _STAGE_LABELS[job.stage] || {};
        _setProgress(
          true,
          label.title || "Processing…",
          label.msg || job.detail || "Working…",
          job.detail || ""
        );
      },
      900,
      300_000
    );

    // Step 3: refresh document list from registry
    const docs = await CitadelAPI.getDocuments(state.currentSessionId);
    state.sources = docs.map((d) => ({
      id: d.id,
      name: d.filename || d.name,
      size: d.size || _fmtSize(d.file_size_bytes || 0),
      active: d.status === "ready",
      status: d.status,
      chunk_count: d.chunk_count || 0,
      content: d.preview_text || d.content || "",
      file_type: d.file_type,
    }));

    renderSources();
    const session = state.sessions.find((item) => item.id === state.currentSessionId);
    if (
      session &&
      session.title === "Untitled Notebook" &&
      docs.length > 0
    ) {
      const firstName = docs[0].filename || docs[0].name || "Notebook";
      const generatedTitle = firstName.replace(/\.[^.]+$/, "").trim() || "Notebook";
      await CitadelAPI.upsertSession(session.id, generatedTitle, session.emoji);
      session.title = generatedTitle;
      if (currentSessionTitle) currentSessionTitle.textContent = generatedTitle;
      renderHistory();
    }
    if (state.sources.length > 0) {
      openPassageReader(state.sources.length - 1);
    }

    _setProgress(true, "Ready!", "Your documents are on your desk.", "");
    await _sleep(700);

  } catch (err) {
    _setProgress(true, "Upload failed", "We couldn't process those files.", err.message);
    await _sleep(1200);
    console.warn("Upload error:", err);
  } finally {
    _setProgress(false);
    state.uploadInProgress = false;
    if (fileInputEl) fileInputEl.value = "";
  }
}

function _setProgress(visible, title = "", msg = "", detail = "") {
  if (!uploadProgressOverlay) return;
  if (!visible) { uploadProgressOverlay.classList.add("hidden"); return; }
  uploadProgressOverlay.classList.remove("hidden");
  if (uploadProgressTitle)   uploadProgressTitle.textContent   = title;
  if (uploadProgressMessage) uploadProgressMessage.textContent = msg;
  if (uploadProgressDetail)  uploadProgressDetail.textContent  = detail;
}
function _sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

// ─── Study Tools Panel ────────────────────────────────────────────────────────

async function runStudyTool(tool) {
  if (state.queryInProgress) return;

  const activeDocs = state.sources.filter((s) => s.active);
  const docIds = activeDocs.map((d) => d.id);

  if (docIds.length === 0) {
    alert("Select at least one document before using study tools.");
    return;
  }

  state.queryInProgress = true;
  hideAmbientWave();

  // Ensure session
  if (!state.currentSessionId) await createNewNotebookAndOpen();
  const session = state.sessions.find((s) => s.id === state.currentSessionId);

  const label = {
    flashcards: "Generate flashcards",
    quiz: "Generate quiz questions",
    summary: "Summarize documents",
    key_terms: "Extract key terms",
    explain_simple: "Explain simply",
  }[tool] || tool;

  if (!state.messages[state.currentSessionId]) state.messages[state.currentSessionId] = [];
  state.messages[state.currentSessionId].push({ role: "user", text: `[Study Tool: ${label}]` });
  if (session) session.messages = state.messages[state.currentSessionId];

  sendBtnEl.disabled = true;
  renderChat();

  try {
    const res = await CitadelAPI.runStudyTool(tool, docIds);
    const aiMsg = { role: "ai", text: res?.answer || "Could not generate study content." };
    state.messages[state.currentSessionId].push(aiMsg);
    if (session) session.messages = state.messages[state.currentSessionId];
    try { await CitadelAPI.addMessage(session.id, "ai", aiMsg.text); } catch { /* best effort */ }
  } catch (err) {
    state.messages[state.currentSessionId].push({ role: "ai", text: `⚠️ Study tool error: ${err.message}` });
  }

  state.queryInProgress = false;
  sendBtnEl.disabled = false;
  renderChat();
}

// ─── Event wiring ─────────────────────────────────────────────────────────────

function setupEvents() {
  // Hub Navigation
  document.getElementById("createNotebookBtn")?.addEventListener("click", () => createNewNotebookAndOpen());
  document.getElementById("backToHubBrand")?.addEventListener("click", returnToHub);
  document.getElementById("topBackToHubBtn")?.addEventListener("click", returnToHub);
  hubSearchInput?.addEventListener("input", (e) => renderHub(e.target.value));

  // Hub Filter Tabs
  document.querySelectorAll(".hub-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".hub-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const filter = tab.dataset.filter;
      document.getElementById("featuredSection")?.classList.toggle("hidden", filter === "my");
      document.getElementById("recentSection")?.classList.toggle("hidden", filter === "featured");
    });
  });

  // Chat send
  sendBtnEl?.addEventListener("click", sendUserMessage);
  promptInputEl?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendUserMessage(); }
  });
  promptInputEl?.addEventListener("input", () => {
    promptInputEl.style.height = "auto";
    promptInputEl.style.height = `${Math.min(promptInputEl.scrollHeight, 150)}px`;
    if (promptInputEl.value.trim().length > 0) hideAmbientWave();
  });

  // Quick prompt chips
  document.querySelectorAll(".starter-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      promptInputEl.value = pill.dataset.prompt;
      hideAmbientWave();
      promptInputEl.focus();
    });
  });

  // File upload
  dropzoneEl?.addEventListener("click", () => fileInputEl?.click());
  quickAttachBtn?.addEventListener("click", () => fileInputEl?.click());
  fileInputEl?.addEventListener("change", (e) => handleFileUploads(e.target.files));
  dropzoneEl?.addEventListener("dragover", (e) => { e.preventDefault(); dropzoneEl.classList.add("dragover"); });
  dropzoneEl?.addEventListener("dragleave", () => dropzoneEl.classList.remove("dragover"));
  dropzoneEl?.addEventListener("drop", (e) => {
    e.preventDefault(); dropzoneEl.classList.remove("dragover"); handleFileUploads(e.dataTransfer.files);
  });

  // Include all toggle
  document.getElementById("toggleAllSourcesBtn")?.addEventListener("click", () => {
    const allChecked = state.sources.every((s) => s.active);
    state.sources.forEach((s) => { if (s.status === "ready") s.active = !allChecked; });
    renderSources();
  });

  // Study tool pills
  document.querySelectorAll(".study-tool-pill").forEach((pill) => {
    pill.addEventListener("click", () => runStudyTool(pill.dataset.tool));
  });

  // New chat
  document.getElementById("newChatBtn")?.addEventListener("click", () => createNewNotebookAndOpen());
  currentSessionTitle?.addEventListener("click", renameCurrentNotebook);

  // Sidebar / drawer toggles
  document.getElementById("sidebarToggleBtn")?.addEventListener("click", () => {
    const sidebar = document.getElementById("sidebar");
    sidebar?.classList.toggle("collapsed");
    workspaceView?.classList.toggle("sidebar-collapsed", sidebar?.classList.contains("collapsed"));
  });
  document.getElementById("drawerToggleBtn")?.addEventListener("click", () => {
    const isClosed = inspectorDrawer?.classList.toggle("closed") ?? true;
    workspaceView?.classList.toggle("drawer-closed", isClosed);
  });
  document.getElementById("closeDrawerBtn")?.addEventListener("click", () => {
    inspectorDrawer?.classList.add("closed");
    workspaceView?.classList.add("drawer-closed");
  });

  // Reader tabs
  document.querySelectorAll(".reader-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".reader-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const target = tab.dataset.tab;
      _activateReaderTab(target);
    });
  });

  // Theme swatches
  document.querySelectorAll(".swatch-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const selectedTheme = btn.dataset.theme;
      const selectedName = btn.dataset.name;
      document.querySelectorAll(".swatch-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.theme === selectedTheme);
      });
      document.documentElement.setAttribute("data-theme", selectedTheme);
      if (themeNameLabel) themeNameLabel.textContent = selectedName;
    });
  });
}

function _syncResponsivePanels() {
  const sidebar = document.getElementById("sidebar");
  if (!workspaceView || !sidebar || !inspectorDrawer) return;

  if (window.innerWidth <= 980) {
    sidebar.classList.add("collapsed");
    workspaceView.classList.remove("sidebar-collapsed");
  } else {
    sidebar.classList.remove("collapsed");
    workspaceView.classList.remove("sidebar-collapsed");
  }

  if (window.innerWidth <= 1280) {
    inspectorDrawer.classList.add("closed");
    workspaceView.classList.add("drawer-closed");
  } else {
    inspectorDrawer.classList.remove("closed");
    workspaceView.classList.remove("drawer-closed");
  }
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function _fmtSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function _formatDate(iso) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleDateString("en-GB", {
      day: "numeric", month: "short", year: "numeric",
    });
  } catch {
    return iso.slice(0, 10);
  }
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", initStudyWorkspace);