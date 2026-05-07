/* ===== State ===== */
let documents = [];
let conversationHistory = [];
const MAX_HISTORY_TURNS = 10; // keep last 10 user+assistant pairs (20 messages)

/* ===== Helpers ===== */
const $ = id => document.getElementById(id);

function ext(name) {
  const i = name.lastIndexOf('.');
  return i >= 0 ? name.slice(i + 1) : 'txt';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function mdToHtml(text) {
  return text
    .replace(/```([\w]*)\n?([\s\S]*?)```/g, (_, lang, code) =>
      `<pre><code>${escapeHtml(code)}</code></pre>`)
    .replace(/`([^`\n]+)`/g, (_, c) => `<code>${escapeHtml(c)}</code>`)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h3 style="font-size:14px;font-weight:500;margin:10px 0 4px">$1</h3>')
    .replace(/^## (.+)$/gm, '<h2 style="font-size:15px;font-weight:500;margin:12px 0 6px">$1</h2>')
    .replace(/^# (.+)$/gm, '<h2 style="font-size:16px;font-weight:500;margin:12px 0 6px">$1</h2>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/^/, '<p>').replace(/$/, '</p>');
}

/* ===== Banner dismissal ===== */
$('banner-close')?.addEventListener('click', () => {
  document.querySelector('.banner')?.classList.add('hidden');
});

/* ===== Status check ===== */
async function checkStatus() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    if (data.error) {
      $('status-dot').className = 'dot err';
      $('status-text').textContent = 'pinecone error';
    } else {
      $('status-dot').className = 'dot ok';
      $('status-text').textContent = `${data.total_vectors || 0} vectors`;
      $('vec-badge').textContent = `${data.total_vectors || 0} vectors`;
    }
  } catch (e) {
    $('status-dot').className = 'dot err';
    $('status-text').textContent = 'offline';
  }
}

/* ===== Source list ===== */
async function loadSources() {
  try {
    const res = await fetch('/api/sources');
    const data = await res.json();
    documents = data.sources || [];
    renderDocs();
  } catch (e) {
    console.error(e);
  }
}

function renderDocs() {
  const list = $('doc-list');
  list.innerHTML = '';
  if (!documents.length) {
    list.innerHTML = `<p style="text-align:center;color:var(--muted);font-size:11px;padding:20px;font-family:var(--mono)">loading knowledge base...</p>`;
    return;
  }
  documents.forEach(d => {
    const div = document.createElement('div');
    div.className = 'doc-item';
    div.innerHTML = `
      <div class="doc-icon">${escapeHtml(ext(d.source).slice(0, 4))}</div>
      <div class="doc-meta">
        <div class="doc-name" title="${escapeHtml(d.source)}">${escapeHtml(d.source)}</div>
        <div class="doc-sub">${d.chunk_count} chunks</div>
      </div>
    `;
    list.appendChild(div);
  });
}

/* ===== Chat ===== */
function appendMsg(role, html, sources) {
  const area = $('chat-area');
  $('empty-state')?.remove();
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const av = role === 'user' ? 'YOU' : 'DS';
  wrap.innerHTML = `
    <div class="avatar">${av}</div>
    <div class="bubble">${html}</div>
  `;
  if (sources && sources.length) {
    const srcDiv = document.createElement('div');
    srcDiv.className = 'sources';
    const seen = new Map();
    sources.forEach(s => {
      if (!seen.has(s.source) || seen.get(s.source) < s.score) {
        seen.set(s.source, s.score);
      }
    });
    let chipsHtml = '<div class="sources-label">retrieved sources</div><div class="source-chips">';
    [...seen.entries()].forEach(([src, score]) => {
      chipsHtml += `<span class="source-chip">${escapeHtml(src)}<span class="score">${score.toFixed(2)}</span></span>`;
    });
    chipsHtml += '</div>';
    srcDiv.innerHTML = chipsHtml;
    wrap.querySelector('.bubble').appendChild(srcDiv);
  }
  area.appendChild(wrap);
  area.scrollTop = area.scrollHeight;
  return wrap;
}

function appendThinking() {
  const area = $('chat-area');
  const wrap = document.createElement('div');
  wrap.className = 'msg ai';
  wrap.id = 'thinking-msg';
  wrap.innerHTML = `
    <div class="avatar">DS</div>
    <div class="bubble">
      <div class="thinking">
        <div class="dot-anim"><span></span><span></span><span></span></div>
        <span>retrieving &amp; generating</span>
      </div>
    </div>`;
  area.appendChild(wrap);
  area.scrollTop = area.scrollHeight;
}

function removeThinking() { $('thinking-msg')?.remove(); }

async function sendQuery() {
  const q = $('query').value.trim();
  if (!q) return;
  const topK = 5;

  $('query').value = '';
  autoResize($('query'));
  appendMsg('user', escapeHtml(q));

  // Add user message to history BEFORE sending
  conversationHistory.push({ role: 'user', content: q });

  appendThinking();
  $('send-btn').disabled = true;

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: q,
        top_k: topK,
        history: conversationHistory.slice(-MAX_HISTORY_TURNS * 2)
      })
    });

    if (res.status === 429) {
      removeThinking();
      appendMsg('ai', `<div class="error-msg">Rate limit hit (30 queries/min). Wait a moment and try again.</div>`);
      // Roll back user message from history since the request failed
      conversationHistory.pop();
      return;
    }

    const data = await res.json();
    removeThinking();
    if (data.error) {
      appendMsg('ai', `<div class="error-msg">${escapeHtml(data.error)}</div>`);
      conversationHistory.pop();
    } else {
      const answer = data.answer || '(no response)';
      appendMsg('ai', mdToHtml(answer), data.sources || []);
      // Add assistant message to history
      conversationHistory.push({ role: 'assistant', content: answer });
    }
  } catch (e) {
    removeThinking();
    appendMsg('ai', `<div class="error-msg">Network error: ${escapeHtml(e.message)}</div>`);
    conversationHistory.pop();
  } finally {
    $('send-btn').disabled = false;
  }
}

/* ===== New Chat ===== */
function newChat() {
  if (conversationHistory.length === 0) return;
  if (!confirm('Start a new conversation? Current chat will be cleared.')) return;
  conversationHistory = [];
  const area = $('chat-area');
  area.innerHTML = `
    <div class="empty-state" id="empty-state">
      <div class="empty-mark">&#8292;</div>
      <p class="empty-title">Ask anything about VIFHE</p>
      <p class="empty-sub">programs &middot; admissions &middot; fees &middot; policies</p>
    </div>`;
}

/* ===== Wiring ===== */
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

$('send-btn').addEventListener('click', sendQuery);
$('query').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});
$('query').addEventListener('input', e => autoResize(e.target));
$('new-chat-btn')?.addEventListener('click', newChat);

/* ===== Init ===== */
checkStatus();
loadSources();