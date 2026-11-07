import { useRef, useState } from "react";
import { api } from "../api";
import Plot from "./Plot";

export default function DriftView({ onError }) {
  const [result, setResult] = useState(null);
  const [shift, setShift] = useState(1);
  const fileRef = useRef();
  const wrap = async (fn) => { onError(""); try { setResult(await fn()); } catch (e) { onError(e.message); } };

  return (
    <>
      <div className="callout"><b>Data drift detector.</b> If today's customers differ from the training data, causal estimates may no longer hold. Check a new batch before you trust new prescriptions.</div>
      <div className="card">
        <h2>Check a new customer batch</h2>
        <div className="row">
          <button className="ghost" onClick={() => wrap(() => api.driftDemo(0))}>Demo: same behaviour</button>
          <label className="row">Shift <input type="number" step="0.25" min="0" max="2" value={shift} onChange={(e) => setShift(e.target.value)} style={{ width: 80 }} /></label>
          <button className="primary" onClick={() => wrap(() => api.driftDemo(shift))}>Demo: changed behaviour</button>
          <button className="ghost" onClick={() => fileRef.current.click()}>Upload CSV</button>
          <input ref={fileRef} type="file" accept=".csv" hidden onChange={(e) => e.target.files[0] && wrap(() => api.driftUpload(e.target.files[0]))} />
        </div>
      </div>
      {result && (
        <div className={`callout ${result.status === "stable" ? "" : "error"}`}>
          <b>{result.status === "stable" ? "Stable." : "Drift detected."}</b> {result.message}
        </div>
      )}
      {result && (
        <div className="card">
          <h2>Population Stability Index by feature</h2>
          <Plot
            data={[{ type: "bar", x: result.features.map(f => f.feature), y: result.features.map(f => f.psi),
                     marker: { color: result.features.map(f => f.drifted ? "#dc2626" : "#16a34a") } }]}
            layout={{ yaxis: { title: "PSI (above 0.10 = shift)" }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.1, y1: 0.1, line: { dash: "dash", color: "#d97706" } }] }}
          />
        </div>
      )}
    </>
  );
}
