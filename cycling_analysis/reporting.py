import json
from pathlib import Path
import numpy as np

def _svg_plot(path, series, title, ylabel):
    width, height = 900, 340
    vectors = [np.asarray(y, dtype=float) for _, y in series]
    finite = np.concatenate([v[np.isfinite(v)] for v in vectors])
    low, high = float(finite.min()), float(finite.max())
    if high == low: high = low + 1
    colors = ["#0f766e", "#e8793e", "#4f46e5"]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><text x="50" y="30" font-size="20" font-family="sans-serif">{title}</text><line x1="55" y1="290" x2="870" y2="290" stroke="#666"/><line x1="55" y1="50" x2="55" y2="290" stroke="#666"/><text x="10" y="45" font-size="12" font-family="sans-serif">{ylabel}</text>']
    for i, (label, values) in enumerate(series):
        points = " ".join(f"{55+j*815/max(1,len(values)-1):.1f},{290-(v-low)*235/(high-low):.1f}" for j,v in enumerate(values) if np.isfinite(v))
        color = colors[i % len(colors)]
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{650+i*75}" y="45" fill="{color}" font-size="12" font-family="sans-serif">{label}</text>')
    lines.append('</svg>')
    path.write_text(''.join(lines))

def write_report(result, output_dir, charts):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    chart_names = []
    for name, series, title, ylabel in charts:
        _svg_plot(out / name, series, title, ylabel)
        chart_names.append(name)
    result["charts"] = chart_names
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
