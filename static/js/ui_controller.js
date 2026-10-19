/**
 * Citadel Human-Centric UI Controller + Silk Ribbon Wave Stage (static/js/ui_controller.js)
 */

const state = {
  sources: [
    {
      id: "doc-1",
      name: "Project_Overview.md",
      size: "12 KB",
      active: true,
      content:
        "Citadel is a calm, personal study workspace.\n\n1. Upload your class readings, research papers, or markdown notes.\n2. Keep only the files you need checked on your desk.\n3. Ask questions naturally in the writing box below.",
    },
    {
      id: "doc-2",
      name: "Lecture_Notes_Week4.txt",
      size: "8 KB",
      active: true,
      content:
        "Human-Computer Interaction (HCI) Principle: Interfaces should feel warm, predictable, and reduce cognitive load rather than overwhelming the person using them.",
    },
  ],
  sessions: [
    {
      id: "page-1",
      title: "Morning Reading & Notes",
      messages: [
        {
          role: "ai",
          text: "Welcome to your study desk. I've got **Project_Overview.md** and **Lecture_Notes_Week4.txt** open and ready. [Source 1] [Source 2]\n\nDrop any new PDFs or notes on the left, or write what's on your mind below.",
        },
      ],
    },
    {
      id: "page-2",
      title: "HCI Design Principles",
      messages: [
        {
          role: "user",
          text: "What makes an interface feel good to use?",
        },
        {
          role: "ai",
          text: "According to your lecture notes, a great interface reduces visual clutter, uses warm typography, and keeps the core actions effortless to reach. [Source 2]",
        },
      ],
    },
  ],
  currentSessionId: "page-1",
  pinnedNotes: [
    "HCI Principle: Interfaces should feel warm, predictable, and reduce cognitive load.",
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

function initStudyWorkspace() {
  renderSources();
  renderHistory();
  renderChat();
  renderNotes();
  openPassageReader(0);
  setupEvents();
  initSilkWaveAnimation();
}

// ================= SILK RIBBON WAVE ANIMATION =================
function initSilkWaveAnimation() {
  if (!waveCanvas) return;
  const ctx = waveCanvas.getContext("2d");
  let width = 0;
  let height = 0;
  let tick = 0;

  function resize() {
    const rect = waveCanvas.parentElement.getBoundingClientRect();
    width = waveCanvas.width = rect.width * window.devicePixelRatio;
    height = waveCanvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }

  window.addEventListener("resize", resize);
  resize();

  function drawSilkRibbons() {
    const w = waveCanvas.clientWidth;
    const h = waveCanvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    const activeTheme =
      document.documentElement.getAttribute("data-theme") || "coral-lemonade";
    const [colorA, colorB] =
      state.waveColors[activeTheme] || state.waveColors["coral-lemonade"];

    const totalLines = 34; // Fine mesh lines like your uploaded wave image
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

function renderSources() {
  sourceListEl.innerHTML = "";
  const activeDocs = state.sources.filter((s) => s.active);

  activeContextCounter.textContent =
    activeDocs.length === 0
      ? "No documents selected"
      : `${activeDocs.length} document${activeDocs.length === 1 ? "" : "s"} on desk`;

  promptSourcesBar.innerHTML = activeDocs.length
    ? activeDocs
        .map((s) => `<span class="doc-pill">📄 ${s.name}</span>`)
        .join("")
    : `<span class="doc-pill">No files checked</span>`;

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
      <span class="doc-size">${doc.size}</span>
    `;

    const checkbox = card.querySelector(".doc-checkbox");
    checkbox.addEventListener("change", (e) => {
      e.stopPropagation();
      doc.active = e.target.checked;
      renderSources();
    });

    card.addEventListener("click", (e) => {
      if (e.target.tagName !== "INPUT") {
        openPassageReader(index);
      }
    });

    sourceListEl.appendChild(card);
  });
}

function renderHistory() {
  historyListEl.innerHTML = "";
  state.sessions.forEach((sess) => {
    const item = document.createElement("div");
    item.className = `page-item ${
      sess.id === state.currentSessionId ? "active" : ""
    }`;
    item.textContent = sess.title;
    item.addEventListener("click", () => {
      state.currentSessionId = sess.id;
      currentSessionTitle.textContent = sess.title;
      renderHistory();
      renderChat();
    });
    historyListEl.appendChild(item);
  });
}

function renderChat() {
  const session = state.sessions.find((s) => s.id === state.currentSessionId);
  if (!session) return;

  chatFeedEl.innerHTML = `
    <div class="welcome-banner">
      <h2>${getTimeOfDayGreeting()} Ready to think through your notes?</h2>
      <p>Ask anything about the documents sitting on your desk.</p>
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
  noteCountBadge.textContent = state.pinnedNotes.length;
  notesListEl.innerHTML = state.pinnedNotes
    .map((n) => `<div class="highlight-card">${n}</div>`)
    .join("");
}

function formatFriendlyText(text) {
  return text
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
  if (!doc) return;

  inspectorDrawer.classList.remove("closed");
  inspectorContent.innerHTML = `
    <span class="source-badge-meta">SOURCE #${docIndex + 1} • ${doc.size}</span>
    <h3 class="reader-doc-heading">${doc.name}</h3>
    <div class="reader-doc-sheet">${doc.content}</div>
  `;
}

async function sendUserMessage() {
  const text = promptInputEl.value.trim();
  if (!text) return;

  // Hide the animated wave once the conversation starts
  hideAmbientWave();

  const session = state.sessions.find((s) => s.id === state.currentSessionId);
  session.messages.push({ role: "user", text });

  if (session.messages.length === 2 && session.title === "Untitled Page") {
    session.title = text.slice(0, 30) + (text.length > 30 ? "..." : "");
    currentSessionTitle.textContent = session.title;
    renderHistory();
  }

  promptInputEl.value = "";
  promptInputEl.style.height = "auto";
  renderChat();

  const activeDocs = state.sources.filter((s) => s.active);
  const res = await CitadelAPI.queryKnowledgeBase(text, activeDocs);

  session.messages.push({ role: "ai", text: res.answer });
  renderChat();
}

async function handleFileUploads(files) {
  if (!files || !files.length) return;
  const result = await CitadelAPI.uploadDocuments(files);
  if (result && result.sources) {
    state.sources.push(...result.sources);
    renderSources();
    openPassageReader(state.sources.length - 1);
  }
}

function setupEvents() {
  sendBtnEl.addEventListener("click", sendUserMessage);
  promptInputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendUserMessage();
    }
  });

  // Hide the moving background wave the moment the user starts typing
  promptInputEl.addEventListener("input", () => {
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

  dropzoneEl.addEventListener("click", () => fileInputEl.click());
  quickAttachBtn.addEventListener("click", () => fileInputEl.click());
  fileInputEl.addEventListener("change", (e) => handleFileUploads(e.target.files));

  dropzoneEl.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzoneEl.classList.add("dragover");
  });
  dropzoneEl.addEventListener("dragleave", () =>
    dropzoneEl.classList.remove("dragover")
  );
  dropzoneEl.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzoneEl.classList.remove("dragover");
    handleFileUploads(e.dataTransfer.files);
  });

  document.getElementById("toggleAllSourcesBtn").addEventListener("click", () => {
    const allChecked = state.sources.every((s) => s.active);
    state.sources.forEach((s) => (s.active = !allChecked));
    renderSources();
  });

  // Clicking "New Page" brings back the moving wave until they start typing again
  document.getElementById("newChatBtn").addEventListener("click", () => {
    const newId = `page-${Date.now()}`;
    state.sessions.unshift({
      id: newId,
      title: "Untitled Page",
      messages: [
        {
          role: "ai",
          text: "Fresh page opened. Which documents on your desk would you like to explore?",
        },
      ],
    });
    state.currentSessionId = newId;
    currentSessionTitle.textContent = "Untitled Page";
    showAmbientWave();
    renderHistory();
    renderChat();
  });

  document.getElementById("sidebarToggleBtn").addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("collapsed");
  });

  document.getElementById("drawerToggleBtn").addEventListener("click", () => {
    inspectorDrawer.classList.toggle("closed");
  });

  document.getElementById("closeDrawerBtn").addEventListener("click", () => {
    inspectorDrawer.classList.add("closed");
  });

  document.querySelectorAll(".reader-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".reader-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const target = tab.dataset.tab;
      document
        .getElementById("inspectorTab")
        .classList.toggle("hidden", target !== "inspector");
      document
        .getElementById("notesTab")
        .classList.toggle("hidden", target !== "notes");
    });
  });

  // 6-Theme Swatch Selector
  document.querySelectorAll(".swatch-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".swatch-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const selectedTheme = btn.dataset.theme;
      const selectedName = btn.dataset.name;
      document.documentElement.setAttribute("data-theme", selectedTheme);
      if (themeNameLabel) themeNameLabel.textContent = selectedName;
    });
  });
}

document.addEventListener("DOMContentLoaded", initStudyWorkspace);