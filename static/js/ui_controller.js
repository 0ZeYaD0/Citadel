/**
 * Citadel Human-Centric UI Controller + NotebookLM Welcome Hub (static/js/ui_controller.js)
 * Static documents and static chat history have been completely removed.
 */

const state = {
  sources: [],
  sessions: [],
  currentSessionId: null,
  pinnedNotes: [],

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

// DOM References
const homeHubView = document.getElementById("homeHubView");
const workspaceView = document.getElementById("workspaceView");
const featuredGrid = document.getElementById("featuredGrid");
const recentGrid = document.getElementById("recentGrid");
const hubSearchInput = document.getElementById("hubSearchInput");

const sourceListEl = document.getElementById("sourceList");
const historyListEl = document.getElementById("historyList");
const chatFeedEl = document.getElementById("chatFeed");
const promptInputEl = document.getElementById("promptInput");
const sendBtnEl = document.getElementById("sendBtn");
const dropzoneEl = document.getElementById("dropzone");
const fileInputEl = document.getElementById("fileInput");
const quickAttachBtn = document.getElementById("quickAttachBtn");
const promptSourcesBar = document.getElementById("promptSourcesBar");
const activeContextCounter = document.getElementById("activeContextCounter");
const currentSessionTitle = document.getElementById("currentSessionTitle");
const inspectorDrawer = document.getElementById("inspectorDrawer");
const inspectorContent = document.getElementById("inspectorContent");
const notesListEl = document.getElementById("notesList");
const noteCountBadge = document.getElementById("noteCountBadge");
const themeNameLabel = document.getElementById("themeNameLabel");
const ambientWaveStage = document.getElementById("ambientWaveStage");
const waveCanvas = document.getElementById("waveCanvas");
const uploadProgressOverlay = document.getElementById("uploadProgressOverlay");
const uploadProgressTitle = document.getElementById("uploadProgressTitle");
const uploadProgressMessage = document.getElementById("uploadProgressMessage");
const uploadProgressDetail = document.getElementById("uploadProgressDetail");

function initStudyWorkspace() {
  renderHub();
  renderSources();
  renderHistory();
  renderNotes();
  setupEvents();
  initSilkWaveAnimation();
}

// ================= VIEW NAVIGATION (HUB <-> WORKSPACE) =================
function openWorkspaceWithSession(sessionId) {
  state.currentSessionId = sessionId;
  const session = state.sessions.find((s) => s.id === sessionId);
  if (session && currentSessionTitle) {
    currentSessionTitle.textContent = session.title;
  }

  homeHubView?.classList.add("hidden");
  workspaceView?.classList.remove("hidden");

  showAmbientWave();
  renderSources();
  renderHistory();
  renderChat();
}

function createNewNotebookAndOpen(customTitle = "Untitled Notebook", emoji = "📓") {
  const newId = `nb-${Date.now()}`;
  const today = new Date().toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });

  state.sessions.unshift({
    id: newId,
    title: customTitle,
    emoji: emoji,
    date: today,
    messages: [],
  });

  openWorkspaceWithSession(newId);
}

function returnToHub() {
  workspaceView?.classList.add("hidden");
  homeHubView?.classList.remove("hidden");
  renderHub(hubSearchInput ? hubSearchInput.value : "");
}

// ================= RENDER WELCOME HUB =================
function renderHub(searchFilter = "") {
  const query = searchFilter.trim().toLowerCase();

  // 1. Render Featured Notebooks
  if (featuredGrid) {
    const filteredFeatured = state.featuredNotebooks.filter(
      (f) =>
        f.title.toLowerCase().includes(query) ||
        f.author.toLowerCase().includes(query)
    );

    featuredGrid.innerHTML = filteredFeatured
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
      </div>
    `
      )
      .join("");

    featuredGrid.querySelectorAll(".featured-card").forEach((card) => {
      card.addEventListener("click", () => {
        createNewNotebookAndOpen(card.dataset.title, "✨");
      });
    });
  }

  // 2. Render Recent User Notebooks (+ New Notebook Tile first)
  if (recentGrid) {
    const filteredSessions = state.sessions.filter((s) =>
      s.title.toLowerCase().includes(query)
    );

    const newTileHTML = `
      <div class="recent-card new-card-tile" id="newNotebookTile">
        <div class="new-tile-plus">+</div>
        <strong>Create new notebook</strong>
      </div>
    `;

    const sessionCardsHTML = filteredSessions
      .map(
        (sess) => `
      <div class="recent-card" data-id="${sess.id}">
        <div class="recent-card-top">
          <span class="recent-emoji">${sess.emoji || "📓"}</span>
          <button class="delete-notebook-btn" data-del="${sess.id}" title="Delete notebook">✕</button>
        </div>
        <div class="recent-card-bottom">
          <h4>${sess.title}</h4>
          <span class="recent-card-meta">${sess.date} • ${state.sources.length} source${
          state.sources.length === 1 ? "" : "s"
        }</span>
        </div>
      </div>
    `
      )
      .join("");

    recentGrid.innerHTML = newTileHTML + sessionCardsHTML;

    const newTile = document.getElementById("newNotebookTile");
    if (newTile) {
      newTile.addEventListener("click", () => createNewNotebookAndOpen());
    }

    recentGrid.querySelectorAll(".recent-card[data-id]").forEach((card) => {
      card.addEventListener("click", (e) => {
        if (e.target.classList.contains("delete-notebook-btn")) return;
        openWorkspaceWithSession(card.dataset.id);
      });
    });

    recentGrid.querySelectorAll(".delete-notebook-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const delId = btn.dataset.del;
        state.sessions = state.sessions.filter((s) => s.id !== delId);
        renderHub(hubSearchInput ? hubSearchInput.value : "");
        renderHistory();
      });
    });
  }
}

// ================= SILK RIBBON WAVE ANIMATION =================
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
      if (w && h && waveCanvas.width !== w * (window.devicePixelRatio || 1)) {
        resize();
      }
      ctx.clearRect(0, 0, w, h);

      const activeTheme =
        document.documentElement.getAttribute("data-theme") || "sunset-horizon";
      const [colorA, colorB] =
        state.waveColors[activeTheme] || state.waveColors["sunset-horizon"];

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

          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
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
  if (ambientWaveStage && !ambientWaveStage.classList.contains("vanished")) {
    ambientWaveStage.classList.add("vanished");
  }
}

function showAmbientWave() {
  if (ambientWaveStage) {
    ambientWaveStage.classList.remove("vanished");
  }
}

function getTimeOfDayGreeting() {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning.";
  if (hour < 18) return "Good afternoon.";
  return "Good evening.";
}

// ================= RENDER SOURCES & HISTORY =================
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
      ? activeDocs
          .map((s) => `<span class="doc-pill">📄 ${s.name}</span>`)
          .join("")
      : `<span class="doc-pill">No files on desk yet</span>`;
  }

  if (state.sources.length === 0) {
    sourceListEl.innerHTML = `<div class="empty-shelf-note">No documents uploaded yet.</div>`;
    return;
  }

  state.sources.forEach((doc, index) => {
    const card = document.createElement("div");
    card.className = "doc-card";
    card.innerHTML = `
      <div class="doc-card-left">
        <input type="checkbox" class="doc-checkbox" ${
          doc.active ? "checked" : ""
        } />
        <span class="doc-title" title="${doc.name}">${doc.name}</span>
      </div>
      <div class="doc-card-actions">
        <span class="doc-size">${doc.size || ""}</span>
        ${
          doc.id
            ? `<button class="doc-delete-btn" type="button" title="Remove document">×</button>`
            : ""
        }
      </div>
    `;

    const checkbox = card.querySelector(".doc-checkbox");
    checkbox.addEventListener("change", (e) => {
      e.stopPropagation();
      doc.active = e.target.checked;
      renderSources();
    });

    const deleteButton = card.querySelector(".doc-delete-btn");
    deleteButton?.addEventListener("click", async (e) => {
      e.stopPropagation();
      try {
        if (typeof CitadelAPI !== "undefined" && CitadelAPI.deleteDocument) {
          await CitadelAPI.deleteDocument(doc.id);
        }
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

function renderHistory() {
  if (!historyListEl) return;
  historyListEl.innerHTML = "";

  if (state.sessions.length === 0) {
    historyListEl.innerHTML = `<div class="empty-shelf-note">No past pages yet.</div>`;
    return;
  }

  state.sessions.forEach((sess) => {
    const item = document.createElement("div");
    item.className = `page-item ${
      sess.id === state.currentSessionId ? "active" : ""
    }`;
    item.textContent = sess.title;
    item.addEventListener("click", () => {
      state.currentSessionId = sess.id;
      if (currentSessionTitle) currentSessionTitle.textContent = sess.title;
      renderHistory();
      renderChat();
    });
    historyListEl.appendChild(item);
  });
}

function renderChat() {
  if (!chatFeedEl) return;
  const session = state.sessions.find((s) => s.id === state.currentSessionId);
  if (!session) return;

  chatFeedEl.innerHTML = `
    <div class="welcome-banner">
      <h2>${getTimeOfDayGreeting()} Ready to think through your notes?</h2>
      <p>Upload your reading material on the left or write a question below.</p>
    </div>
  `;

  session.messages.forEach((msg) => {
    const card = document.createElement("div");
    card.className = `entry-card ${msg.role}`;

    if (msg.role === "user") {
      card.textContent = msg.text;
    } else {
      card.innerHTML = `
        <div class="entry-meta">
          <span class="entry-author">Citadel Companion</span>
          <button class="bookmark-btn" title="Save this thought">Save note</button>
        </div>
        <div class="entry-body">${formatFriendlyText(msg.text)}</div>
      `;

      const saveBtn = card.querySelector(".bookmark-btn");
      saveBtn.addEventListener("click", () => {
        state.pinnedNotes.unshift(msg.text.replace(/\[Source \d+\]/g, ""));
        renderNotes();
        saveBtn.textContent = "✓ Saved";
      });

      card.querySelectorAll(".citation-tag").forEach((tag) => {
        tag.addEventListener("click", () => {
          const idx = parseInt(tag.dataset.index, 10) - 1;
          openPassageReader(idx);
        });
      });
    }

    chatFeedEl.appendChild(card);
  });

  chatFeedEl.scrollTop = chatFeedEl.scrollHeight;
}

function renderNotes() {
  if (noteCountBadge) noteCountBadge.textContent = state.pinnedNotes.length;
  if (!notesListEl) return;

  if (state.pinnedNotes.length === 0) {
    notesListEl.innerHTML = `<div class="reader-empty"><p>No saved highlights yet. Click "Save note" on any response to pin it here.</p></div>`;
    return;
  }
  notesListEl.innerHTML = state.pinnedNotes
    .map((n) => `<div class="highlight-card">${n}</div>`)
    .join("");
}

function formatFriendlyText(text) {
  return (text || "")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.*?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>")
    .replace(
      /\[Source (\d+)\]/g,
      `<span class="citation-tag" data-index="$1">Ref $1</span>`
    );
}

function openPassageReader(docIndex) {
  const doc = state.sources[docIndex];
  if (!doc || !inspectorDrawer || !inspectorContent) return;

  inspectorDrawer.classList.remove("closed");
  inspectorContent.innerHTML = `
    <span class="source-badge-meta">SOURCE #${docIndex + 1} • ${doc.size || ""}</span>
    <h3 class="reader-doc-heading">${doc.name}</h3>
    <div class="reader-doc-sheet">${doc.content || ""}</div>
  `;
}

async function sendUserMessage() {
  const text = promptInputEl.value.trim();
  if (!text) return;

  hideAmbientWave();

  let session = state.sessions.find((s) => s.id === state.currentSessionId);
  if (!session) {
    createNewNotebookAndOpen();
    session = state.sessions[0];
  }

  session.messages.push({ role: "user", text });

  // Automatically name the notebook after the user's first question
  if (session.messages.length === 1 && session.title === "Untitled Notebook") {
    session.title = text.slice(0, 32) + (text.length > 32 ? "..." : "");
    if (currentSessionTitle) currentSessionTitle.textContent = session.title;
    renderHistory();
  }

  promptInputEl.value = "";
  promptInputEl.style.height = "auto";
  renderChat();

  const activeDocs = state.sources.filter((s) => s.active);
  try {
    const res = await CitadelAPI.queryKnowledgeBase(text, activeDocs);
    session.messages.push({
      role: "ai",
      text: res?.answer || "No response received from the model.",
    });
  } catch (err) {
    session.messages.push({
      role: "ai",
      text: `⚠️ Error communicating with the AI model: ${err.message}`,
    });
  }
  renderChat();
}

async function handleFileUploads(files) {
  if (!files || !files.length) return;
  if (document.body.classList.contains("uploading-documents")) return;

  const fileCount = files.length;
  document.body.classList.add("uploading-documents");
  if (uploadProgressOverlay) uploadProgressOverlay.classList.remove("hidden");
  if (uploadProgressTitle) {
    uploadProgressTitle.textContent =
      fileCount === 1 ? "Adding your resource" : `Adding ${fileCount} resources`;
  }
  if (uploadProgressMessage) {
    uploadProgressMessage.textContent = "Reading your documents...";
  }
  if (uploadProgressDetail) {
    uploadProgressDetail.textContent = `${fileCount} file${
      fileCount === 1 ? "" : "s"
    } selected`;
  }

  const stageTimer = window.setTimeout(() => {
    if (uploadProgressMessage) {
      uploadProgressMessage.textContent = "Creating searchable notes...";
    }
    if (uploadProgressDetail) {
      uploadProgressDetail.textContent =
        "Splitting passages and building local embeddings";
    }
  }, 650);

  try {
    const result = await CitadelAPI.uploadDocuments(files);
    if (result && result.sources) {
      state.sources.push(...result.sources);
      renderSources();
      openPassageReader(state.sources.length - 1);
      if (uploadProgressMessage) {
        uploadProgressMessage.textContent = "Your resources are ready.";
      }
      if (uploadProgressDetail) {
        uploadProgressDetail.textContent = "You can now ask questions about them.";
      }
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
  } catch (error) {
    if (uploadProgressMessage) {
      uploadProgressMessage.textContent = "We couldn't add those resources.";
    }
    if (uploadProgressDetail) {
      uploadProgressDetail.textContent = "Please check the files and try again.";
    }
    await new Promise((resolve) => window.setTimeout(resolve, 900));
    console.warn("Document upload failed:", error.message);
  } finally {
    window.clearTimeout(stageTimer);
    if (uploadProgressOverlay) uploadProgressOverlay.classList.add("hidden");
    document.body.classList.remove("uploading-documents");
    if (fileInputEl) fileInputEl.value = "";
  }
}

function setupEvents() {
  // Hub Navigation Buttons
  document.getElementById("createNotebookBtn")?.addEventListener("click", () => {
    createNewNotebookAndOpen();
  });
  document.getElementById("backToHubBrand")?.addEventListener("click", returnToHub);
  document.getElementById("topBackToHubBtn")?.addEventListener("click", returnToHub);

  if (hubSearchInput) {
    hubSearchInput.addEventListener("input", (e) => {
      renderHub(e.target.value);
    });
  }

  // Hub Filter Tabs (All / My notebooks / Featured)
  document.querySelectorAll(".hub-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".hub-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const filter = tab.dataset.filter;
      const featSec = document.getElementById("featuredSection");
      const recSec = document.getElementById("recentSection");
      if (featSec) featSec.classList.toggle("hidden", filter === "my");
      if (recSec) recSec.classList.toggle("hidden", filter === "featured");
    });
  });

  // Workspace Prompt & Send
  sendBtnEl?.addEventListener("click", sendUserMessage);
  promptInputEl?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendUserMessage();
    }
  });

  promptInputEl?.addEventListener("input", () => {
    promptInputEl.style.height = "auto";
    promptInputEl.style.height = `${Math.min(promptInputEl.scrollHeight, 150)}px`;

    if (promptInputEl.value.trim().length > 0) {
      hideAmbientWave();
    }
  });

  document.querySelectorAll(".starter-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      promptInputEl.value = pill.dataset.prompt;
      hideAmbientWave();
      promptInputEl.focus();
    });
  });

  dropzoneEl?.addEventListener("click", () => fileInputEl?.click());
  quickAttachBtn?.addEventListener("click", () => fileInputEl?.click());
  fileInputEl?.addEventListener("change", (e) => handleFileUploads(e.target.files));

  dropzoneEl?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzoneEl.classList.add("dragover");
  });
  dropzoneEl?.addEventListener("dragleave", () =>
    dropzoneEl.classList.remove("dragover")
  );
  dropzoneEl?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzoneEl.classList.remove("dragover");
    handleFileUploads(e.dataTransfer.files);
  });

  document.getElementById("toggleAllSourcesBtn")?.addEventListener("click", () => {
    const allChecked = state.sources.every((s) => s.active);
    state.sources.forEach((s) => (s.active = !allChecked));
    renderSources();
  });

  document.getElementById("newChatBtn")?.addEventListener("click", () => {
    createNewNotebookAndOpen();
  });

  document.getElementById("sidebarToggleBtn")?.addEventListener("click", () => {
    document.getElementById("sidebar")?.classList.toggle("collapsed");
  });

  document.getElementById("drawerToggleBtn")?.addEventListener("click", () => {
    inspectorDrawer?.classList.toggle("closed");
  });

  document.getElementById("closeDrawerBtn")?.addEventListener("click", () => {
    inspectorDrawer?.classList.add("closed");
  });

  document.querySelectorAll(".reader-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".reader-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const target = tab.dataset.tab;
      document
        .getElementById("inspectorTab")
        ?.classList.toggle("hidden", target !== "inspector");
      document
        .getElementById("notesTab")
        ?.classList.toggle("hidden", target !== "notes");
    });
  });

  // Sync Theme Swatches across both Hub and Workspace
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

document.addEventListener("DOMContentLoaded", initStudyWorkspace);