/**
 * Citadel API Client  (static/js/api_client.js)
 * All network calls to the Python backend go through this module.
 */

const CitadelAPI = {
  baseUrl: "", // Same origin when served by app.py

  // ─── Documents ──────────────────────────────────────────────────────────

  /** Upload files; returns { job_id, document_ids, rejected } immediately. */
  async uploadDocuments(files) {
    const formData = new FormData();
    Array.from(files).forEach((file) => formData.append("files", file));
    const sessionId = window.CitadelCurrentSessionId;
    if (sessionId) formData.append("session_id", sessionId);

    const response = await fetch(`${this.baseUrl}/api/upload`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Upload failed (${response.status}): ${detail}`);
    }
    return await response.json();
  },

  /** Poll a background job until it reaches done/error. Calls onProgress(job) each tick. */
  async pollJob(jobId, onProgress, intervalMs = 800, maxWaitMs = 300_000) {
    const deadline = Date.now() + maxWaitMs;
    return new Promise((resolve, reject) => {
      const tick = async () => {
        try {
          const job = await this.getJob(jobId);
          if (onProgress) onProgress(job);
          if (job.status === "done" || job.status === "error") {
            resolve(job);
            return;
          }
          if (Date.now() > deadline) {
            reject(new Error("Upload timed out waiting for processing."));
            return;
          }
          setTimeout(tick, intervalMs);
        } catch (err) {
          reject(err);
        }
      };
      tick();
    });
  },

  /** Fetch one job's status. */
  async getJob(jobId) {
    const res = await fetch(`${this.baseUrl}/api/jobs/${encodeURIComponent(jobId)}`);
    if (!res.ok) throw new Error(`Job ${jobId} not found`);
    return await res.json();
  },

  /** Fetch the document registry (persisted across refreshes). */
  async getDocuments(sessionId = null) {
    try {
      const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
      const res = await fetch(`${this.baseUrl}/api/documents${query}`);
      if (!res.ok) throw new Error("Registry unavailable");
      const data = await res.json();
      return data.documents || [];
    } catch {
      return [];
    }
  },

  /** Delete one document from the vector store and registry. */
  async deleteDocument(documentId) {
    const response = await fetch(
      `${this.baseUrl}/api/documents/${encodeURIComponent(documentId)}`,
      { method: "DELETE" }
    );
    if (!response.ok) throw new Error("Document deletion failed");
    return await response.json();
  },

  /** Delete ALL documents. */
  async deleteAllDocuments() {
    const response = await fetch(`${this.baseUrl}/api/documents`, { method: "DELETE" });
    if (!response.ok) throw new Error("Failed to delete all documents");
    return await response.json();
  },

  // ─── Query ──────────────────────────────────────────────────────────────

  /** Send a RAG query; returns { answer, citations, evidence_found, … }. */
  async queryKnowledgeBase(query, activeSources, minSimilarity = null) {
    try {
      const body = {
        query,
        sources: activeSources.map((s) => ({ id: s.id, name: s.name })),
      };
      if (minSimilarity !== null) body.min_similarity = minSimilarity;

      const response = await fetch(`${this.baseUrl}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        const detail = errorBody.detail || `Server returned ${response.status}`;
        if (response.status === 503)
          return { answer: "The RAG pipeline is warming up — please try again in a moment." };
        if (response.status === 502)
          return { answer: "There was a problem generating an answer. Try rephrasing." };
        throw new Error(detail);
      }
      return await response.json();
    } catch (err) {
      console.warn("Query endpoint unavailable:", err.message);
      return { answer: `⚠️ Could not reach the backend: ${err.message}` };
    }
  },

  // ─── Study Tools ─────────────────────────────────────────────────────────

  /** Generate flashcards / quiz / summary / key_terms / explain_simple. */
  async runStudyTool(tool, documentIds = [], chapter = null, count = 5) {
    const body = { tool, document_ids: documentIds, count };
    if (chapter !== null) body.chapter = chapter;

    const response = await fetch(`${this.baseUrl}/api/study-tools`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || "Study tool failed");
    }
    return await response.json();
  },

  // ─── Sessions ────────────────────────────────────────────────────────────

  /** Fetch all persisted sessions. */
  async getSessions() {
    try {
      const res = await fetch(`${this.baseUrl}/api/sessions`);
      if (!res.ok) return [];
      const data = await res.json();
      return data.sessions || [];
    } catch {
      return [];
    }
  },

  /** Create or update a session. */
  async upsertSession(sessionId, title, emoji = "📓") {
    const url = sessionId
      ? `${this.baseUrl}/api/sessions?session_id=${encodeURIComponent(sessionId)}`
      : `${this.baseUrl}/api/sessions`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, emoji }),
    });
    if (!res.ok) throw new Error("Failed to save session");
    return await res.json();
  },

  /** Delete a session and its messages. */
  async deleteSession(sessionId) {
    const res = await fetch(
      `${this.baseUrl}/api/sessions/${encodeURIComponent(sessionId)}`,
      { method: "DELETE" }
    );
    if (!res.ok) throw new Error("Failed to delete session");
    return await res.json();
  },

  /** Fetch messages for one session. */
  async getMessages(sessionId) {
    try {
      const res = await fetch(
        `${this.baseUrl}/api/sessions/${encodeURIComponent(sessionId)}/messages`
      );
      if (!res.ok) return [];
      const data = await res.json();
      return data.messages || [];
    } catch {
      return [];
    }
  },

  /** Append a message to a session. */
  async addMessage(sessionId, role, text) {
    const res = await fetch(
      `${this.baseUrl}/api/sessions/${encodeURIComponent(sessionId)}/messages`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role, text }),
      }
    );
    if (!res.ok) throw new Error("Failed to save message");
    return await res.json();
  },

  // ─── Notes ───────────────────────────────────────────────────────────────

  /** Fetch all notes, optionally for one session. */
  async getNotes(sessionId = null) {
    try {
      const url = sessionId
        ? `${this.baseUrl}/api/notes?session_id=${encodeURIComponent(sessionId)}`
        : `${this.baseUrl}/api/notes`;
      const res = await fetch(url);
      if (!res.ok) return [];
      const data = await res.json();
      return data.notes || [];
    } catch {
      return [];
    }
  },

  /** Persist a note. Returns { note_id }. */
  async saveNote(text, sessionId = null, documentId = null, pageNumber = null, tags = []) {
    const res = await fetch(`${this.baseUrl}/api/notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        session_id: sessionId,
        document_id: documentId,
        page_number: pageNumber,
        tags,
      }),
    });
    if (!res.ok) throw new Error("Failed to save note");
    return await res.json();
  },

  /** Delete one note. */
  async deleteNote(noteId) {
    const res = await fetch(`${this.baseUrl}/api/notes/${noteId}`, { method: "DELETE" });
    if (!res.ok) throw new Error("Failed to delete note");
    return await res.json();
  },

  // ─── Health ───────────────────────────────────────────────────────────────

  async healthCheck() {
    try {
      const res = await fetch(`${this.baseUrl}/health`);
      return await res.json();
    } catch {
      return { status: "unreachable" };
    }
  },
};