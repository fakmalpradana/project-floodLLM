import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

// ─── Config ────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_URL || ''

// ─── Helpers ───────────────────────────────────────────────────────────────
const STATUS_LABELS = {
  parsing_prompt: 'Parsing prompt…',
  geocoding: 'Geocoding location…',
  downloading_satellite_data: 'Downloading satellite imagery…',
  downloading_rainfall: 'Downloading rainfall data…',
  processing_flood_detection: 'Running flood detection…',
  validating_with_optical: 'Validating with optical imagery…',
  generating_risk_assessment: 'Generating risk assessment…',
  generating_map: 'Building interactive map…',
  generating_report: 'Generating report…',
  completed: 'Complete',
  failed: 'Failed',
  processing: 'Processing…',
}

function riskColor(level) {
  if (level >= 0.7) return 'text-red-400'
  if (level >= 0.4) return 'text-yellow-400'
  return 'text-green-400'
}

function riskLabel(level) {
  if (level >= 0.7) return 'HIGH'
  if (level >= 0.4) return 'MODERATE'
  return 'LOW'
}

// ─── Sub-components ────────────────────────────────────────────────────────

function ProgressBar({ progress, status }) {
  const isError = status === 'failed'
  const bar = isError ? 'bg-red-500' : 'bg-blue-500'
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-slate-400 mb-1">
        <span>{STATUS_LABELS[status] || status}</span>
        <span>{progress}%</span>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
        <div
          className={`${bar} h-2 rounded-full transition-all duration-500`}
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  )
}

function ResultCard({ job }) {
  const res = job.result || {}
  const risk = res.risk_level ?? 0

  return (
    <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
      <Stat label="Flood Area" value={`${(res.flood_area_km2 ?? 0).toFixed(2)} km²`} />
      <Stat label="Hectares" value={`${(res.flood_area_ha ?? 0).toFixed(1)} ha`} />
      <Stat
        label="Risk Level"
        value={riskLabel(risk)}
        valueClass={riskColor(risk)}
      />
      <Stat label="Risk Score" value={(risk).toFixed(2)} />
    </div>
  )
}

function Stat({ label, value, valueClass = 'text-white' }) {
  return (
    <div className="bg-slate-800 rounded-lg p-3 text-center">
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className={`text-lg font-bold ${valueClass}`}>{value}</p>
    </div>
  )
}

function JobCard({ job, onSelect, isSelected }) {
  const statusDot = {
    completed: 'bg-green-400',
    failed: 'bg-red-400',
    processing: 'bg-blue-400 animate-pulse',
    parsing_prompt: 'bg-blue-400 animate-pulse',
    geocoding: 'bg-blue-400 animate-pulse',
    downloading_satellite_data: 'bg-blue-400 animate-pulse',
    downloading_rainfall: 'bg-blue-400 animate-pulse',
    processing_flood_detection: 'bg-blue-400 animate-pulse',
    validating_with_optical: 'bg-blue-400 animate-pulse',
    generating_risk_assessment: 'bg-blue-400 animate-pulse',
    generating_map: 'bg-blue-400 animate-pulse',
    generating_report: 'bg-blue-400 animate-pulse',
  }[job.status] || 'bg-slate-400'

  return (
    <button
      onClick={() => onSelect(job.job_id)}
      className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors text-sm ${
        isSelected
          ? 'bg-blue-600 text-white'
          : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
      }`}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${statusDot}`} />
        <span className="font-mono text-xs opacity-60">{job.job_id}</span>
      </div>
      <p className="truncate">{job.prompt}</p>
    </button>
  )
}

function MapViewer({ jobId }) {
  const mapUrl = `${API_BASE}/api/map/${jobId}`
  return (
    <div className="mt-4 rounded-xl overflow-hidden border border-slate-700 bg-slate-900">
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700">
        <span className="text-sm text-slate-400">Interactive Flood Map</span>
        <a
          href={mapUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
        >
          Open full map ↗
        </a>
      </div>
      <iframe
        src={mapUrl}
        title="Flood Map"
        className="w-full h-[420px] border-0"
      />
    </div>
  )
}

// ─── Main App ──────────────────────────────────────────────────────────────
export default function App() {
  const [prompt, setPrompt] = useState('')
  const [jobs, setJobs] = useState([])           // [{job_id, status, progress, prompt, result}]
  const [selectedJobId, setSelectedJobId] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const pollingRef = useRef({})                   // { [jobId]: intervalId }

  // ── Poll a single job ──────────────────────────────────────────────────
  const pollJob = useCallback((jobId) => {
    if (pollingRef.current[jobId]) return         // already polling

    const id = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status/${jobId}`)
        if (!res.ok) return
        const data = await res.json()

        setJobs((prev) =>
          prev.map((j) => (j.job_id === jobId ? { ...j, ...data } : j))
        )

        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(pollingRef.current[jobId])
          delete pollingRef.current[jobId]
        }
      } catch (_) {/* network hiccup — keep polling */}
    }, 2000)

    pollingRef.current[jobId] = id
  }, [])

  // ── Restore jobs from localStorage on mount ──────────────────────────
  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem('floodllm_jobs') || '[]')
    if (saved.length) {
      setJobs(saved)
      setSelectedJobId(saved[0].job_id)
      saved.forEach((j) => {
        if (j.status !== 'completed' && j.status !== 'failed') pollJob(j.job_id)
      })
    }
  }, [pollJob])

  // ── Persist jobs to localStorage ─────────────────────────────────────
  useEffect(() => {
    localStorage.setItem('floodllm_jobs', JSON.stringify(jobs))
  }, [jobs])

  // ── Cleanup intervals on unmount ─────────────────────────────────────
  useEffect(() => {
    return () => Object.values(pollingRef.current).forEach(clearInterval)
  }, [])

  // ── Submit a new prompt ───────────────────────────────────────────────
  async function handleSubmit(e) {
    e.preventDefault()
    const text = prompt.trim()
    if (!text) return

    setSubmitting(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/api/prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text }),
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()

      const newJob = {
        job_id: data.job_id,
        status: data.status,
        progress: 0,
        prompt: text,
        result: null,
      }
      setJobs((prev) => [newJob, ...prev])
      setSelectedJobId(data.job_id)
      setPrompt('')
      pollJob(data.job_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  // ── Clear all jobs ────────────────────────────────────────────────────
  function clearHistory() {
    Object.values(pollingRef.current).forEach(clearInterval)
    pollingRef.current = {}
    setJobs([])
    setSelectedJobId(null)
  }

  const selectedJob = jobs.find((j) => j.job_id === selectedJobId)

  // ─── Render ─────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col">
      {/* ── Header ── */}
      <header className="border-b border-slate-800 px-6 py-4 flex items-center gap-3">
        <svg className="w-7 h-7 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
            d="M3 15a4 4 0 004 4h10a4 4 0 000-8 4 4 0 00-7.75-2M3 15H1m4 0H3" />
        </svg>
        <h1 className="text-xl font-bold tracking-tight">FloodLLM</h1>
        <span className="text-xs text-slate-500 ml-1">AI Flood Monitoring</span>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* ── Sidebar: Job History ── */}
        <aside className="w-72 flex-shrink-0 border-r border-slate-800 flex flex-col">
          <div className="p-4 border-b border-slate-800 flex items-center justify-between">
            <span className="text-sm font-medium text-slate-300">Job History</span>
            {jobs.length > 0 && (
              <button
                onClick={clearHistory}
                className="text-xs text-slate-500 hover:text-red-400 transition-colors"
              >
                Clear all
              </button>
            )}
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {jobs.length === 0 && (
              <p className="text-xs text-slate-600 text-center mt-8">
                No jobs yet.<br />Submit a prompt to get started.
              </p>
            )}
            {jobs.map((job) => (
              <JobCard
                key={job.job_id}
                job={job}
                onSelect={setSelectedJobId}
                isSelected={job.job_id === selectedJobId}
              />
            ))}
          </div>
        </aside>

        {/* ── Main panel ── */}
        <main className="flex-1 flex flex-col overflow-y-auto">
          {/* Prompt form */}
          <div className="p-6 border-b border-slate-800">
            <form onSubmit={handleSubmit} className="space-y-3">
              <label className="block text-sm font-medium text-slate-300 mb-1">
                Natural Language Flood Query
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit(e)
                }}
                placeholder='e.g. "Show flood extent in Jakarta for the last 7 days" or "Assess flood risk in Bangkok this week"'
                rows={3}
                className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              />
              <div className="flex items-center gap-3">
                <button
                  type="submit"
                  disabled={submitting || !prompt.trim()}
                  className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                >
                  {submitting ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
                      </svg>
                      Submitting…
                    </>
                  ) : 'Analyze Flood'}
                </button>
                <span className="text-xs text-slate-500">⌘ + Enter to submit</span>
              </div>
              {error && (
                <p className="text-sm text-red-400 bg-red-950/40 border border-red-800 rounded-lg px-3 py-2">
                  {error}
                </p>
              )}
            </form>
          </div>

          {/* Job detail */}
          <div className="flex-1 p-6">
            {!selectedJob ? (
              <div className="h-full flex flex-col items-center justify-center text-center text-slate-600">
                <svg className="w-16 h-16 mb-4 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                    d="M3 15a4 4 0 004 4h10a4 4 0 000-8 4 4 0 00-7.75-2M3 15H1m4 0H3" />
                </svg>
                <p className="text-sm">Submit a flood query above to begin analysis.</p>
              </div>
            ) : (
              <div>
                {/* Job header */}
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-xs font-mono text-slate-500 mb-1">
                      Job ID: {selectedJob.job_id}
                    </p>
                    <p className="text-base text-slate-200">{selectedJob.prompt}</p>
                  </div>
                  {selectedJob.status === 'completed' && (
                    <a
                      href={`${API_BASE}/api/report/${selectedJob.job_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="ml-4 flex-shrink-0 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-xs text-slate-300 transition-colors"
                    >
                      Download Report ↓
                    </a>
                  )}
                </div>

                {/* Progress */}
                {selectedJob.status !== 'completed' && (
                  <ProgressBar
                    progress={selectedJob.progress}
                    status={selectedJob.status}
                  />
                )}

                {/* Error */}
                {selectedJob.status === 'failed' && selectedJob.error && (
                  <div className="mt-4 bg-red-950/40 border border-red-800 rounded-lg px-4 py-3 text-sm text-red-300">
                    <strong>Error:</strong> {selectedJob.error}
                  </div>
                )}

                {/* Results */}
                {selectedJob.status === 'completed' && selectedJob.result && (
                  <>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-2 h-2 rounded-full bg-green-400" />
                      <span className="text-sm text-green-400 font-medium">Analysis complete</span>
                    </div>
                    <ResultCard job={selectedJob} />
                    <MapViewer jobId={selectedJob.job_id} />
                  </>
                )}

                {/* Parsed metadata (debug) */}
                {selectedJob.parsed && (
                  <details className="mt-6">
                    <summary className="text-xs text-slate-500 cursor-pointer hover:text-slate-400">
                      Parsed query parameters
                    </summary>
                    <pre className="mt-2 text-xs bg-slate-800/60 rounded-lg p-3 overflow-x-auto text-slate-400">
                      {JSON.stringify(selectedJob.parsed, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
