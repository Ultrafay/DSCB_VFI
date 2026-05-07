/* ===== State ===== */
let documents = [];

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

/* ===== Banner dismissal (in-memory only; resets per page load) ===== */
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
    list.innerHTML = `<p style="text-align:center;color:var(--muted);font-size:11px;padding:20px;font-family:var(--mono)">no documents yet</p>`;
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
      <div class="doc-del" title="Delete">×</div>
    `;
    div.querySelector('.doc-del').addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${d.source}" from your vector store?`)) return;
      const res = await fetch('/api/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: d.source })
      });
      if (res.status === 429) {
        alert('Rate limit hit — slow down a bit.');
        return;
      }
      const r = await res.json();
      if (r.deleted >= 0) {
        await loadSources();
        await checkStatus();
      }
    });
    list.appendChild(div);
  });
}

/* ===== Upload ===== */
async function uploadFiles(files) {
  if (!files.length) return;

  const progress = $('upload-progress');
  const items = [];
  for (const f of files) {
    const item = document.createElement('div');
    item.className = 'upload-item pending';
    item.innerHTML = `<div class="upload-spinner"></div><span>${escapeHtml(f.name)}</span>`;
    progress.appendChild(item);
    items.push(item);
  }

  const fd = new FormData();
  for (const f of files) fd.append('files', f);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });

    if (res.status === 429) {
      items.forEach(i => {
        i.className = 'upload-item error';
        i.innerHTML = '<span>✗ rate limit hit (8 uploads / hour)</span>';
        setTimeout(() => i.remove(), 4000);
      });
      return;
    }
    if (res.status === 413) {
      items.forEach(i => {
        i.className = 'upload-item error';
        i.innerHTML = '<span>✗ upload too large (max 20MB total)</span>';
        setTimeout(() => i.remove(), 4000);
      });
      return;
    }

    const data = await res.json();
    (data.results || []).forEach((r, idx) => {
      const item = items[idx];
      if (r.error) {
        item.className = 'upload-item error';
        item.innerHTML = `<span>✗ ${escapeHtml(r.filename)}: ${escapeHtml(r.error)}</span>`;
      } else {
        item.className = 'upload-item success';
        const trunc = r.truncated ? ' (truncated)' : '';
        item.innerHTML = `<span>✓ ${escapeHtml(r.filename)} (${r.chunks} chunks${trunc})</span>`;
      }
      setTimeout(() => item.remove(), 4000);
    });
    await loadSources();
    await checkStatus();
    $('empty-state')?.remove();
  } catch (e) {
    items.forEach(i => {
      i.className = 'upload-item error';
      i.innerHTML = `<span>✗ upload failed: ${escapeHtml(e.message)}</span>`;
    });
  }
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
  const topK = parseInt($('topk').value) || 5;

  $('query').value = '';
  autoResize($('query'));
  appendMsg('user', escapeHtml(q));
  appendThinking();
  $('send-btn').disabled = true;

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: topK })
    });

    if (res.status === 429) {
      removeThinking();
      appendMsg('ai', `<div class="error-msg">Rate limit hit (30 queries/min). Wait a moment and try again.</div>`);
      return;
    }

    const data = await res.json();
    removeThinking();
    if (data.error) {
      appendMsg('ai', `<div class="error-msg">${escapeHtml(data.error)}</div>`);
    } else {
      appendMsg('ai', mdToHtml(data.answer || '(no response)'), data.sources || []);
    }
  } catch (e) {
    removeThinking();
    appendMsg('ai', `<div class="error-msg">Network error: ${escapeHtml(e.message)}</div>`);
  } finally {
    $('send-btn').disabled = false;
  }
}

/* ===== Wiring ===== */
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

const dropZone = $('drop-zone');
const fileInput = $('file-input');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  uploadFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener('change', () => {
  uploadFiles([...fileInput.files]);
  fileInput.value = '';
});

$('send-btn').addEventListener('click', sendQuery);
$('query').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});
$('query').addEventListener('input', e => autoResize(e.target));

$('refresh-btn').addEventListener('click', async () => {
  await loadSources();
  await checkStatus();
});
$('clear-btn').addEventListener('click', async () => {
  if (!confirm('Delete all YOUR documents from the vector store? This cannot be undone.')) return;
  const res = await fetch('/api/clear', { method: 'POST' });
  if (res.status === 429) {
    alert('Rate limit hit on clear endpoint.');
    return;
  }
  await loadSources();
  await checkStatus();
});

/* ===== Init ===== */
checkStatus();
loadSources();
