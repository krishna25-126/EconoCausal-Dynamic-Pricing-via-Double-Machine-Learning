import Plotly from "plotly.js-basic-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const BasePlot = createPlotlyComponent(Plotly);

// Shared look: transparent background so charts follow the page theme.
export default function Plot({ data, layout = {}, height = 320 }) {
  const fg = getComputedStyle(document.documentElement).getPropertyValue("--text").trim() || "#1d2433";
  return (
    <BasePlot
      data={data}
      layout={{
        height, margin: { l: 55, r: 15, t: 30, b: 45 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: fg }, legend: { orientation: "h", y: -0.2 }, ...layout,
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%" }}
      useResizeHandler
    />
  );
}