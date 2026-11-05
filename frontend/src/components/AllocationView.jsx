import { useState } from "react";
import { api } from "../api";
import Plot from "./Plot";

const money = (x) => `${x < 0 ? "-" : ""}$${Math.abs(x).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

export default function AllocationView({ budget, setBudget, alloc, setAlloc, onError }) {
  const [busy, setBusy] = useState(false);
  const [user, setUser] = useState(null);
  const [query, setQuery] = useState("");

  const run = async () => {
    setBusy(true); onError("");
    try { setAlloc(await api.allocate(Number(budget))); setUser(null); } catch (e) { onError(e.message); } finally { setBusy(false); }
  };
  const open = async (id) => { try { setUser(await api.userIte(id)); } catch (e) { onError(e.message); } };
  const rows = alloc ? alloc.table.filter(r => !query || String(r.user_id).includes(query)) : [];
  const gainKey = alloc && "true_profit_gain" in alloc.strategies[0] ? "true_profit_gain" : "expected_profit_gain";

  return (
    <>
      <div className="card">
        <h2>Set the campaign budget</h2>
        <div className="row">
          <span>$</span>
          <input type="number" min="100" step="500" value={budget} onChange={(e) => setBudget(e.target.value)} />
          <button className="primary" onClick={run} disabled={busy}>{busy ? "Optimizing…" : "Allocate discounts"}</button>
        </div>
      </div>

      {alloc && (
        <>
          <div className="callout"><b>{alloc.summary}</b></div>
          <div className="kpis">
            <div className="kpi"><div className="v">{alloc.customers_discounted.toLocaleString()} / {alloc.customers.toLocaleString()}</div><div className="l">Customers discounted</div></div>
            <div className="kpi"><div className="v">{money(alloc.expected_spend)}</div><div className="l">Expected spend of {money(alloc.budget)}</div></div>
            <div className="kpi"><div className="v">{alloc.segments.Persuadable?.toLocaleString() || 0}</div><div className="l">Persuadables</div></div>
            <div className="kpi"><div className="v">{alloc.segments["Sleeping dog"] || 0}</div><div className="l">Sleeping dogs (do not discount)</div></div>
          </div>

          <div className="grid g2">
            <div className="card">
              <h2>Profit gain by strategy (same budget)</h2>
              <Plot
                data={[{ type: "bar", x: alloc.strategies.map(s => s.strategy), y: alloc.strategies.map(s => s[gainKey]),
                         marker: { color: alloc.strategies.map(s => s[gainKey] >= 0 ? "#4f46e5" : "#dc2626") },
                         text: alloc.strategies.map(s => money(s[gainKey])), textposition: "outside" }]}
                layout={{ yaxis: { title: "Extra profit ($)" } }}
              />
              <p className="muted">Blanket and predictive targeting spend money on customers who would have bought anyway.</p>
            </div>
            <div className="card">
              <h2>Discount levels assigned</h2>
              <Plot
                data={[{ type: "bar", x: Object.keys(alloc.discount_counts).map(d => `$${d}`), y: Object.values(alloc.discount_counts), marker: { color: "#16a34a" } }]}
                layout={{ xaxis: { title: "Discount" }, yaxis: { title: "Customers" } }}
              />
            </div>
          </div>

          <div className="card">
            <h2>Allocation matrix (the prescription)</h2>
            <div className="row" style={{ marginBottom: 10 }}>
              <input type="text" placeholder="Search user_id" value={query} onChange={(e) => setQuery(e.target.value)} />
              <a className="ghost" href={api.downloadUrl}>Download full CSV</a>
              <span className="muted">Showing top {alloc.table.length} of {alloc.table_total_rows} by profit gain. Click a row for $10 vs $20 detail.</span>
            </div>
            {user && (
              <div className="callout">
                <b>Customer {user.user_id}</b>: buys with probability {(user.baseline_prob * 100).toFixed(0)}% without a discount.
                {" "}{user.levels.filter(l => [10, 20].includes(l.discount)).map(l => (
                  <span key={l.discount}> With ${l.discount}: <b>{(l.prob * 100).toFixed(0)}%</b> ({l.lift >= 0 ? "+" : ""}{(l.lift * 100).toFixed(1)} pts, 90% range {(l.lift_low * 100).toFixed(1)} to {(l.lift_high * 100).toFixed(1)}).</span>
                ))}
              </div>
            )}
            <div className="tablewrap">
              <table>
                <thead><tr><th>user_id</th><th>Segment</th><th>Buy prob. now</th><th>Lift if $10</th><th>Lift if $20</th><th>Assigned discount</th><th>Expected profit gain</th></tr></thead>
                <tbody>
                  {rows.map(r => (
                    <tr key={r.user_id} className="click" onClick={() => open(r.user_id)}>
                      <td>{r.user_id}</td>
                      <td><span className={`tag ${r.segment.split(" ")[0]}`}>{r.segment}</span></td>
                      <td>{(r.baseline_prob * 100).toFixed(0)}%</td>
                      <td>{r.lift_if_10 >= 0 ? "+" : ""}{(r.lift_if_10 * 100).toFixed(1)} pts</td>
                      <td>{r.lift_if_20 >= 0 ? "+" : ""}{(r.lift_if_20 * 100).toFixed(1)} pts</td>
                      <td><b>${r.discount}</b></td>
                      <td>{money(r.expected_profit_gain)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  );
}
