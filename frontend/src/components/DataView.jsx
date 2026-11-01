import { useRef, useState } from "react";
import { api } from "../api";

export default function DataView({ info, setInfo, report, setReport, onError, onTrained }) {
  const [busy, setBusy] = useState("");
  const fileRef = useRef();

  const run = async (label, fn) => {
    setBusy(label); onError("");
    try { await fn(); } catch (e) { onError(e.message); } finally { setBusy(""); }
  };
  const loadDemo = () => run("demo", async () => { setReport(null); setInfo(await api.loadDemo(6000)); });
  const upload = (e) => {
    const f = e.target.files[0];
    if (f) run("upload", async () => { setReport(null); setInfo(await api.upload(f)); });
  };
  const train = () => run("train", async () => { const r = await api.train(300); setReport(r); onTrained(); });
  const pct = (x) => `${(x * 100).toFixed(1)}%`;

  return (
    <>
      <div className="card">
        <h2>1. Load historical campaign data</h2>
        <p className="muted">
          One row per customer: <code>user_id, age, income_k, tenure_months, past_purchases, engagement, cart_value,
          discount, purchased, margin</code>. <a href={api.templateUrl}>Download a CSV template</a>.
        </p>
        <div className="row">
          <button className="primary" onClick={loadDemo} disabled={!!busy}>{busy === "demo" ? "Generating…" : "Use demo retail data"}</button>
          <button className="ghost" onClick={() => fileRef.current.click()} disabled={!!busy}>Upload CSV</button>
          <input ref={fileRef} type="file" accept=".csv" hidden onChange={upload} />
        </div>
      </div>

      {info && (
        <div className="card">
          <h2>Data summary</h2>
          <div className="kpis">
            <div className="kpi"><div className="v">{info.rows.toLocaleString()}</div><div className="l">Customers</div></div>
            <div className="kpi"><div className="v">{pct(info.share_discounted)}</div><div className="l">Received a discount</div></div>
            <div className="kpi"><div className="v">{pct(info.purchase_rate_discounted)}</div><div className="l">Bought, discounted</div></div>
            <div className="kpi"><div className="v">{pct(info.purchase_rate_not_discounted)}</div><div className="l">Bought, not discounted</div></div>
          </div>
          <p className="muted">
            A naive reading says discounts barely change buying. That is misleading: past discounts went to customers who were
            already unlikely to buy. Causal ML corrects for this.
          </p>
          <div className="tablewrap">
            <table>
              <thead><tr>{info.columns.filter(c => !c.startsWith("true_")).map(c => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>{info.preview.map((r, i) => <tr key={i}>{info.columns.filter(c => !c.startsWith("true_")).map(c => <td key={c}>{r[c]}</td>)}</tr>)}</tbody>
            </table>
          </div>
        </div>
      )}

      {info && (
        <div className="card">
          <h2>2. Train the causal model</h2>
          <p className="muted">Double Machine Learning removes confounding, then a causal forest estimates each customer's own response to a discount.</p>
          <button className="primary" onClick={train} disabled={!!busy}>{busy === "train" ? "Training…" : report ? "Retrain" : "Train causal model"}</button>
          {report && (
            <div className="kpis" style={{ marginTop: 16 }}>
              <div className="kpi"><div className="v">{(report.average_effect_per_dollar * 100).toFixed(2)} pts</div><div className="l">Avg. lift in buy probability per $1 (causal)</div></div>
              <div className="kpi"><div className="v">{(report.naive_effect_per_dollar * 100).toFixed(2)} pts</div><div className="l">Same, from a naive model (biased)</div></div>
              {report.validation && <div className="kpi"><div className="v">{(report.validation.mean_true * 100).toFixed(2)} pts</div><div className="l">Ground truth (synthetic data only)</div></div>}
            </div>
          )}
        </div>
      )}
    </>
  );
}