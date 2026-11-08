import { useState } from "react";
import DataView from "./components/DataView";
import UpliftView from "./components/UpliftView";
import AllocationView from "./components/AllocationView";
import AuditView from "./components/AuditView";
import DriftView from "./components/DriftView";

const TABS = [
  ["data", "Data & Model"],
  ["uplift", "Causal Uplift"],
  ["allocation", "Allocation"],
  ["audit", "Causal Audit"],
  ["drift", "Drift"],
];

export default function App() {
  const [tab, setTab] = useState("data");
  const [info, setInfo] = useState(null);
  const [report, setReport] = useState(null);
  const [alloc, setAlloc] = useState(null);
  const [budget, setBudget] = useState(5000);
  const [error, setError] = useState("");
  const trained = !!report;

  return (
    <>
      <header>
        <h1>EconoCausal</h1>
        <p>Show who a discount will actually persuade, not just who is likely to buy.</p>
      </header>
      <nav>
        {TABS.map(([id, label]) => (
          <button key={id} className={tab === id ? "active" : ""} disabled={id !== "data" && !trained} onClick={() => setTab(id)}>{label}</button>
        ))}
      </nav>
      <main>
        {error && <div className="callout error"><b>Error:</b> {error}</div>}
        {tab === "data" && <DataView info={info} setInfo={(i) => { setInfo(i); setAlloc(null); }} report={report} setReport={setReport} onError={setError} onTrained={() => setTab("uplift")} />}
        {tab === "uplift" && trained && <UpliftView report={report} />}
        {tab === "allocation" && trained && <AllocationView budget={budget} setBudget={setBudget} alloc={alloc} setAlloc={setAlloc} onError={setError} />}
        {tab === "audit" && trained && <AuditView report={report} onError={setError} />}
        {tab === "drift" && trained && <DriftView onError={setError} />}
      </main>
    </>
  );
}
