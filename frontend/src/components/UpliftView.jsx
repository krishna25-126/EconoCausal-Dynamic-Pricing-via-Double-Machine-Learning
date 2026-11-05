import Plot from "./Plot";

export default function UpliftView({ report }) {
  const q = report.qini, bins = report.ite_bins;
  const mids = report.ite_distribution.map((_, i) => ((bins[i] + bins[i + 1]) / 2) * 100);
  return (
    <>
      <div className="callout">
        <b>Does the causal model beat a random rollout?</b> The Qini curve is measured on a randomized hold-out: treating customers in the
        model's order (blue) should add far more purchases than treating them at random (grey dashed).
      </div>
      <div className="grid g2">
        {q && (
          <div className="card">
            <h2>Qini / uplift curve</h2>
            <Plot
              data={[
                { x: q.fraction.map(f => f * 100), y: q.model, name: "Causal model", mode: "lines", line: { color: "#4f46e5", width: 3 } },
                { x: q.fraction.map(f => f * 100), y: q.random, name: "Random rollout", mode: "lines", line: { color: "#98a2b3", dash: "dash" } },
              ]}
              layout={{ xaxis: { title: "% of customers treated (ranked by model)" }, yaxis: { title: "Incremental purchases" } }}
            />
            <p className="muted">Qini coefficient: <b>{q.qini_coefficient.toFixed(1)}</b> (higher is better; 0 equals random).</p>
          </div>
        )}
        {report.deciles && (
          <div className="card">
            <h2>Observed uplift by decile</h2>
            <Plot
              data={[{ type: "bar", x: report.deciles.map(d => `D${d.decile}`), y: report.deciles.map(d => d.observed_uplift * 100),
                       marker: { color: "#4f46e5" }, name: "Observed uplift" }]}
              layout={{ xaxis: { title: "Decile of predicted effect (D1 = most persuadable)" }, yaxis: { title: "Extra buyers per 100 treated" } }}
            />
            <p className="muted">Uplift should fall from D1 to D10. That proves the ranking tracks real, not just predicted, effects.</p>
          </div>
        )}
        <div className="card">
          <h2>Individual treatment effects ($10 discount)</h2>
          <Plot
            data={[{ type: "bar", x: mids, y: report.ite_distribution, marker: { color: "#16a34a" } }]}
            layout={{ xaxis: { title: "Change in buy probability (percentage points)" }, yaxis: { title: "Customers" }, bargap: 0.05 }}
          />
          <p className="muted">Customers differ widely. Some respond strongly, some not at all, a few react negatively.</p>
        </div>
        <div className="card">
          <h2>Why matching matters</h2>
          <Plot
            data={[
              { type: "bar", orientation: "h", y: report.matching.balance.map(b => b.feature), x: report.matching.balance.map(b => Math.abs(b.smd_before)), name: "Before matching", marker: { color: "#dc2626" } },
              { type: "bar", orientation: "h", y: report.matching.balance.map(b => b.feature), x: report.matching.balance.map(b => Math.abs(b.smd_after)), name: "After matching", marker: { color: "#16a34a" } },
            ]}
            layout={{ barmode: "group", xaxis: { title: "|Standardized mean difference| (below 0.1 = balanced)" }, margin: { l: 110, r: 15, t: 30, b: 50 } }}
          />
        </div>
      </div>
    </>
  );
}
