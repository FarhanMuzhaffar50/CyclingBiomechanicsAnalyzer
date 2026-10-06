import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const PROGRESS = { queued: '10%', extracting_2d_pose: '30%', estimating_3d_pose: '60%', analyzing: '85%', generating_report: '95%', completed: '100%', failed: '100%' }
const STAGES = { queued: 'Queued', extracting_2d_pose: 'Extracting 2D pose', estimating_3d_pose: 'Estimating 3D pose', analyzing: 'Analyzing pedal cycles', generating_report: 'Generating report', completed: 'Analysis complete', failed: 'Analysis failed' }

function metric(result, keys, fallback = '—') {
  for (const key of keys) {
    const value = key.split('.').reduce((obj, part) => obj?.[part], result)
    if (value !== undefined && value !== null) return typeof value === 'number' ? value.toFixed(3) : String(value)
  }
  return fallback
}

function App() {
  const [file, setFile] = useState(null), [job, setJob] = useState(null), [result, setResult] = useState(null), [error, setError] = useState('')
  const [mode, setMode] = useState('video'), [fps, setFps] = useState('60')
  useEffect(() => {
    if (!job || ['completed', 'failed'].includes(job.status)) return undefined
    const timer = setInterval(async () => { try { const response = await fetch(`${API}/api/analyses/${job.analysis_id}`); if (response.ok) setJob(await response.json()) } catch { /* transient network errors are retried */ } }, 1500)
    return () => clearInterval(timer)
  }, [job])
  useEffect(() => {
    if (job?.status !== 'completed') return
    fetch(`${API}/api/analyses/${job.analysis_id}/results`).then(r => r.json()).then(setResult).catch(e => setError(e.message))
  }, [job?.status])
  const charts = useMemo(() => {
    const paths = result?.charts || result?.report?.charts || []
    return Array.isArray(paths) ? paths : []
  }, [result])
  async function submit(event) {
    event.preventDefault(); setError('')
    if (!file) return setError(mode === 'video' ? 'Choose a cycling video first.' : 'Choose an X3D.npy file first.')
    if (mode === 'pose' && (!Number.isFinite(Number(fps)) || Number(fps) <= 0 || Number(fps) > 240)) return setError('Enter an FPS between 0 and 240.')
    const body = new FormData(); body.append(mode === 'video' ? 'video' : 'pose', file)
    if (mode === 'pose') body.append('fps', fps)
    try { const response = await fetch(`${API}${mode === 'video' ? '/api/analyses' : '/api/poses'}`, { method: 'POST', body }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Upload failed'); setJob(data); setResult(null) } catch (e) { setError(e.message) }
  }
  const reset = () => { setFile(null); setJob(null); setResult(null); setError('') }
  return <main>
    <header><div className="eyebrow">MOTION ANALYSIS / PORTFOLIO MVP</div><h1>Cycling <span>Biomechanics</span> Analyzer</h1><p>Upload a cycling video and receive structured motion metrics from markerless 3D pose analysis.</p></header>
    {!job && <form className="upload card" onSubmit={submit}>
      <div className="mode-switch"><button type="button" className={mode === 'video' ? 'active' : ''} onClick={() => { setMode('video'); setFile(null); setError('') }}>Cycling video</button><button type="button" className={mode === 'pose' ? 'active' : ''} onClick={() => { setMode('pose'); setFile(null); setError('') }}>Existing 3D pose</button></div>
      <label className="drop"><input key={mode} type="file" accept={mode === 'video' ? 'video/*' : '.npy'} onChange={e => setFile(e.target.files?.[0] || null)} /><strong>{file ? file.name : mode === 'video' ? 'Choose a cycling video' : 'Choose an X3D.npy file'}</strong><small>{mode === 'video' ? 'MP4, MOV, AVI, MKV, or WebM' : 'MotionBERT H36M-17 pose array'}</small></label>
      {mode === 'pose' && <label className="fps-field">Video FPS <input type="number" min="0.01" max="240" step="any" value={fps} onChange={e => setFps(e.target.value)} /></label>}
      {error && <div className="error">{error}</div>}<button type="submit">Analyze {mode === 'video' ? 'video' : '3D pose'} <span>→</span></button>
    </form>}
    {job && <section className="card workflow"><div className="status-row"><div><div className="eyebrow">ANALYSIS JOB</div><h2>{job.status === 'completed' ? 'Your report is ready' : job.status === 'failed' ? 'Analysis could not finish' : 'Processing your video'}</h2></div><code>{job.analysis_id.slice(0, 8)}</code></div><div className="progress"><div className="progress-fill" style={{width: PROGRESS[job.stage] || PROGRESS[job.status] || '10%'}} /></div><p className="stage">{STAGES[job.stage] || job.stage || STAGES[job.status]}</p>{job.error && <div className="error">{job.error}</div>}{job.status === 'failed' && <button onClick={reset} className="secondary">Try another video</button>}</section>}
    {result && <section className="results"><div className="results-heading"><div><div className="eyebrow">BIOMECHANICAL REPORT</div><h2>Movement summary</h2></div><button className="secondary" onClick={reset}>New analysis</button></div><div className="metrics"><Metric label="Pedal cycles" value={result?.cycles?.count ?? '—'}/><Metric label="Absolute NSI (%)" value={metric(result, ['symmetry.mean_absolute_nsi_percent'])}/><Metric label="Cross-correlation" value={metric(result, ['symmetry.cross_correlation'])}/><Metric label="Pelvis RMS jerk (model units/s³)" value={metric(result, ['stability.pelvis_vertical_rms_jerk_model_units_per_sec3'])}/></div><div className="detail-grid"><div className="card detail"><h3>Kinematics</h3><p>Right knee flexion (mean) <b>{metric(result, ['kinematics.right_knee_flexion_degrees.mean'])}°</b></p><p>Right knee flexion (range) <b>{metric(result, ['kinematics.right_knee_flexion_degrees.min'])}° – {metric(result, ['kinematics.right_knee_flexion_degrees.max'])}°</b></p><p>Angular velocity RMS (deg/s) <b>{metric(result, ['kinematics.right_knee_angular_velocity_rms_deg_per_sec'])}</b></p></div><div className="card detail"><h3>Symmetry & stability</h3><p>Phase lag <b>{metric(result, ['symmetry.phase_lag_seconds'])} s</b></p><p>Cycle consistency (model units) <b>{metric(result, ['cycles.right_knee_y_consistency_model_units'])}</b></p><p>Duration <b>{metric(result, ['metadata.duration_seconds'])} s</b></p></div></div>{charts.length > 0 && <div className="card charts"><h3>Generated charts</h3><div className="chart-grid">{charts.map((chart, i) => { const path = typeof chart === 'string' ? chart : chart.path || chart.url; if (!path) return null; const url = path.startsWith('http') ? path : `${API}/api/analyses/${job.analysis_id}/charts/${path.split('/').pop()}`; return <figure key={i}><img src={url} alt={typeof chart === 'string' ? chart.split('/').pop() : chart.name || `Chart ${i + 1}`} /><figcaption>{typeof chart === 'string' ? chart.split('/').pop() : chart.name || `Chart ${i + 1}`} <a href={url} target="_blank" rel="noreferrer">open ↗</a></figcaption></figure> })}</div></div>}</section>}
    <footer>AlphaPose + MotionBERT inference · Custom biomechanics signal processing</footer>
  </main>
}
function Metric({ label, value }) { return <div className="metric card"><small>{label}</small><strong>{value}</strong></div> }
createRoot(document.getElementById('root')).render(<App />)
