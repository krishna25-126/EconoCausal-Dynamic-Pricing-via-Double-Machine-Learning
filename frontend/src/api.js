// Thin client for the EconoCausal REST API.
export const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

const post = (path, body) =>
  request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });

export const api = {
  health: () => request("/health"),
  dag: () => request("/dag"),
  loadDemo: (n = 6000) => post(`/data/demo?n=${n}`),
  upload: (file) => { const f = new FormData(); f.append("file", file); return request("/data/upload", { method: "POST", body: f }); },
  train: (trees = 300) => post(`/train?trees=${trees}`),
  uplift: () => request("/uplift"),
  audit: () => request("/audit"),
  allocate: (budget, maxRows = 300) => post("/allocate", { budget, max_rows: maxRows }),
  userIte: (id) => request(`/users/${id}/ite`),
  driftDemo: (shift) => post(`/drift/demo?shift=${shift}`),
  driftUpload: (file) => { const f = new FormData(); f.append("file", file); return request("/drift/upload", { method: "POST", body: f }); },
  templateUrl: `${API}/data/template`,
  downloadUrl: `${API}/allocate/download`,
};