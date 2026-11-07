import { useEffect, useState } from "react";
import { api } from "../api";

function Dag({ graph }) {
  const conf = graph.nodes.filter(n => n.role === "confounder");
  const pos = {};
  conf.forEach((n, i) => (pos[n.id] = { x: 90, y: 40 + i * 52 }));
  pos.discount = { x: 330, y: 40 + ((conf.length - 1) * 52) / 2 - 40 };
  pos.purchased = { x: 560, y: pos.discount.y };
  const H = 80 + conf.length * 52;
  const color = { treatment: "#4f46e5", outcome: "#16a34a", confounder: "#98a2b3" };
  return (
    <svg viewBox={`0 0 650 ${H}`} style={{ width: "100%", maxHeight: 420 }}>
      <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#98a2b3" /></marker></defs>
      {graph.edges.map((e, i) => (
        <line key={i} x1={pos[e.source].x + 50} y1={pos[e.source].y} x2={pos[e.target].x - 55} y2={pos[e.target].y}
              stroke={e.source === "discount" ? "#4f46e5" : "#c3c9d4"} strokeWidth={e.source === "discount" ? 3 : 1} markerEnd="url(#arr)" />
      ))}
      {graph.nodes.map(n => (
        <g key={n.id}>
          <rect x={pos[n.id].x - 55} y={pos[n.id].y - 16} width="110" height="32" rx="8" fill="none" stroke={color[n.role]} strokeWidth="2" />
          <text x={pos[n.id].x} y={pos[n.id].y + 5} textAnchor="middle" fontSize="13" fill="currentColor">{n.id}</text>
        </g>
      ))}
    </svg>
  );
}

export default function AuditView({ report, onError }) {
  const [audit, setAudit] = useState(null);
  const [graph, setGraph] = useState(null);
  useEffect(() => {
    api.audit().then(setAudit).catch(e => onError(e.message));
    api.dag().then(setGraph).catch(e => onError(e.message));
  }, []);

  return (
    <>
      <div className="callout">
        <b>Causal audit.</b> We try to break the estimate. If it survives these refutation tests (DoWhy), the effect is unlikely to be
        an artifact of noise or an unmodelled variable.
      </div>
      <div className="card">
        <h2>Refutation tests</h2>
        {!audit ? <p className="muted">Running tests…</p> : (
          <>
            <p>Average effect: <b>{(audit.estimated_effect_per_dollar * 100).toFixed(3)} pts per $1</b>.{" "}
              <span className={audit.all_passed ? "good" : "bad"}>{audit.all_passed ? "All tests passed." : "Some tests failed."}</span></p>
            <div className="tablewrap"><table>
              <thead><tr><th>Test</th><th>What it checks</th><th>Original</th><th>After test</th><th>Result</th></tr></thead>
              <tbody>{audit.tests.map(t => (
                <tr key={t.test}><td>{t.test}</td><td style={{ whiteSpace: "normal" }}>{t.what_it_checks}</td>
                  <td>{t.original_effect.toFixed(5)}</td><td>{t.new_effect.toFixed(5)}</td>
                  <td className={t.passed ? "good" : "bad"}><b>{t.passed ? "PASS" : "FAIL"}</b></td></tr>
              ))}</tbody>
            </table></div>
          </>
        )}
      </div>
      <div className="grid g2">
        <div className="card"><h2>Causal graph (DAG)</h2>{graph ? <Dag graph={graph} /> : null}
          <p className="muted">Customer features influence both who got a discount and who buys. Adjusting for them isolates the discount's own effect.</p></div>
        <div className="card">
          <h2>Propensity matching</h2>
          <p>{report.matching.n_matched_pairs.toLocaleString()} matched pairs of discounted and non-discounted customers.
            Max imbalance fell from <b>{report.matching.max_abs_smd_before.toFixed(2)}</b> to <b>{report.matching.max_abs_smd_after.toFixed(2)}</b>.</p>
          <p className="muted">Matching imitates an A/B test using historical data: matched customers look alike except for the discount.
            Matched effect of getting any discount: {(report.matching.att_matched * 100).toFixed(1)} pts.</p>
        </div>
      </div>
    </>
  );
}
