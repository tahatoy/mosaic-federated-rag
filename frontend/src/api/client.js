/**
 * MOSAIC-v2 Kurumsal API İstemcisi
 */

const API_BASE = '/api';

export async function fetchConfig() {
  try {
    const res = await fetch(`${API_BASE}/config`);
    if (!res.ok) throw new Error(`Config HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('API config fallback:', err);
    return {
      env: 'hq',
      branch_id: 'hq',
      title: 'MOSAIC Kurumsal Finans & Analitik Platformu',
      accent_color: '#2563eb'
    };
  }
}

export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    return {
      status: 'offline',
      services: { rag: 'down', orchestrator: 'down' }
    };
  }
}

export async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) throw new Error(`Stats HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    return { collection_count: 0 };
  }
}

export async function fetchFLStatus() {
  try {
    const res = await fetch(`${API_BASE}/fl_status`);
    if (!res.ok) throw new Error(`FL status HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    return { active: false, connected_branches: 0 };
  }
}

export async function fetchFLAudit() {
  try {
    const res = await fetch(`${API_BASE}/fl_audit`);
    if (!res.ok) throw new Error(`FL audit HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Failed to fetch FL audit data:', err);
    return null;
  }
}

export async function fetchNetworkStatus() {
  try {
    const res = await fetch(`${API_BASE}/network_status`);
    if (!res.ok) throw new Error(`Network status HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('Failed to fetch network status:', err);
    return null;
  }
}

export async function triggerFL(type = 'llm') {
  const res = await fetch(`${API_BASE}/trigger_fl`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type })
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `FL trigger failed HTTP ${res.status}`);
  }
  return await res.json();
}

export async function uploadDocument(file, sourceName = '', branchId = null) {
  const formData = new FormData();
  formData.append('file', file);
  if (sourceName) {
    formData.append('source_name', sourceName);
  }
  if (branchId !== null && branchId !== undefined) {
    formData.append('branch_id', String(branchId));
  }

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Upload error HTTP ${res.status}`);
  }
  return await res.json();
}

/**
 * Stream query response via NDJSON / SSE
 */
export async function streamQuery({ query, branchId = 0, history = [], onChunk, onDone, onError }) {
  try {
    const response = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, branch_id: branchId, history })
    });

    if (!response.ok) {
      throw new Error(`Query HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep last incomplete chunk

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const parsed = JSON.parse(line);
          onChunk(parsed);
        } catch (e) {
          // Plain text fallback
          onChunk({ step: 'token', token: line });
        }
      }
    }

    if (buffer.trim()) {
      try {
        const parsed = JSON.parse(buffer);
        onChunk(parsed);
      } catch (e) {
        onChunk({ step: 'token', token: buffer });
      }
    }

    if (onDone) onDone();
  } catch (err) {
    console.error('Stream error:', err);
    if (onError) onError(err);
  }
}
