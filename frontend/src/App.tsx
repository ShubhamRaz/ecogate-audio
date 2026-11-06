import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid, Legend
} from 'recharts';

interface PredictionRecord {
  filename: string;
  semantic_class: string;
  predicted_bird_category: string;
  confidence: number;
  bird_activity_prob: number;
  novelty_score: number;
  estimated_expert_benefit: number;
  model_used: string;
  expert_invoked: boolean;
  unknown_or_unresolved_status: string;
  inference_time_ms: number;
  resource_usage: {
    compute_cost_units: number;
    memory_kb: number;
    electrical_energy: string;
  };
  timestamp: number;
}

const App: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [latestPrediction, setLatestPrediction] = useState<PredictionRecord | null>(null);
  const [history, setHistory] = useState<PredictionRecord[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const fetchDashboardData = async () => {
    try {
      const [hRes, mRes] = await Promise.all([
        fetch('http://localhost:8000/history'),
        fetch('http://localhost:8000/metrics')
      ]);
      if (hRes.ok) {
        const hData = await hRes.json();
        setHistory(hData.history || []);
      }
      if (mRes.ok) {
        const mData = await mRes.json();
        setMetrics(mData);
      }
    } catch (err) {
      console.error('Backend fetch error:', err);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleUploadAndPredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const res = await fetch('http://localhost:8000/predict/audio', {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data: PredictionRecord = await res.json();
        setLatestPrediction(data);
        fetchDashboardData();
      }
    } finally {
      setLoading(false);
    }
  };

  const classDistData = [
    { name: 'Common Waxbill', count: 25 },
    { name: 'Red-billed Quelea', count: 25 },
    { name: 'Village Weaver', count: 25 },
    { name: 'Yellow-fronted Canary', count: 25 },
    { name: 'Other Birds', count: 25 },
    { name: 'Env. Noise', count: 25 }
  ];

  const modelCompData = metrics?.routing_comparison
    ? Object.entries(metrics.routing_comparison).map(([policy, vals]: [string, any]) => ({
        policy: policy.replace('_routing', ''),
        macroF1: vals.known_species_macro_f1,
        callRate: vals.expert_call_rate,
        loss: vals.expected_decision_loss
      }))
    : [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      <header className="mb-8 border-b border-slate-800 pb-4 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-emerald-400">
            EcoGate-Audio Monitoring Interface
          </h1>
          <p className="text-sm text-slate-400">
            Source-Robust, Cost-Aware Selective Inference for Open-Set Bird Sound Monitoring
          </p>
        </div>
        <span className="px-3 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-700 text-xs">
          Audio-First Selective Pipeline
        </span>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
          <h2 className="text-lg font-semibold mb-3">1. Upload Audio Recording (.wav)</h2>
          <form onSubmit={handleUploadAndPredict} className="space-y-4">
            <input
              type="file"
              accept=".wav"
              onChange={(e) => setSelectedFile(e.target.files ? e.target.files[0] : null)}
              className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-emerald-600 file:text-white"
            />
            <button
              type="submit"
              disabled={!selectedFile || loading}
              className="w-full py-2 px-4 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold disabled:opacity-50"
            >
              {loading ? 'Running Selective Inference...' : 'Run Bird Sound Prediction'}
            </button>
          </form>
        </div>

        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5">
          <h2 className="text-lg font-semibold mb-3">2. Selective Inference Decision & Resource Report</h2>
          {latestPrediction ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <div className="text-slate-400 text-xs">Predicted Category</div>
                <div className="text-base font-bold text-emerald-400 mt-1">
                  {latestPrediction.predicted_bird_category}
                </div>
              </div>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <div className="text-slate-400 text-xs">Calibrated Confidence</div>
                <div className="text-base font-bold mt-1">
                  {(latestPrediction.confidence * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <div className="text-slate-400 text-xs">Model Invoked</div>
                <div className="text-base font-bold text-amber-300 mt-1">
                  {latestPrediction.expert_invoked ? 'Strong Expert' : 'Lightweight Student'}
                </div>
              </div>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <div className="text-slate-400 text-xs">Open-Set / Quality Status</div>
                <div className="text-base font-bold text-sky-300 mt-1">
                  {latestPrediction.unknown_or_unresolved_status}
                </div>
              </div>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <div className="text-slate-400 text-xs">Inference Latency</div>
                <div className="text-base font-bold mt-1">
                  {latestPrediction.inference_time_ms.toFixed(2)} ms
                </div>
              </div>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <div className="text-slate-400 text-xs">Compute Cost Units</div>
                <div className="text-base font-bold mt-1">
                  {latestPrediction.resource_usage.compute_cost_units}x
                </div>
              </div>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <div className="text-slate-400 text-xs">Est. Expert Benefit</div>
                <div className="text-base font-bold mt-1">
                  {latestPrediction.estimated_expert_benefit.toFixed(4)}
                </div>
              </div>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800">
                <div className="text-slate-400 text-xs">Electrical Power</div>
                <div className="text-xs font-mono text-slate-300 mt-1">
                  {latestPrediction.resource_usage.electrical_energy}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-slate-400 text-sm">
              Upload a WAV recording to inspect student vs. expert routing, open-set status, and latency.
            </p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 h-80">
          <h3 className="text-base font-semibold mb-3">Acoustic Class Distribution</h3>
          <ResponsiveContainer width="100%" height="85%">
            <BarChart data={classDistData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip />
              <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 h-80">
          <h3 className="text-base font-semibold mb-3">Routing Policy Macro-F1 vs Decision Loss</h3>
          <ResponsiveContainer width="100%" height="85%">
            <LineChart data={modelCompData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="policy" stroke="#94a3b8" fontSize={10} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="macroF1" stroke="#34d399" strokeWidth={2} />
              <Line type="monotone" dataKey="loss" stroke="#f87171" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <h3 className="text-base font-semibold mb-3">Recent Prediction History</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="border-b border-slate-800 text-slate-400">
              <tr>
                <th className="py-2">File</th>
                <th className="py-2">Semantic Output</th>
                <th className="py-2">Category</th>
                <th className="py-2">Confidence</th>
                <th className="py-2">Model Used</th>
                <th className="py-2">Latency (ms)</th>
                <th className="py-2">Cost Units</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item, idx) => (
                <tr key={idx} className="border-b border-slate-800/50">
                  <td className="py-2 font-mono">{item.filename}</td>
                  <td className="py-2">{item.semantic_class}</td>
                  <td className="py-2 font-semibold text-emerald-400">{item.predicted_bird_category}</td>
                  <td className="py-2">{(item.confidence * 100).toFixed(1)}%</td>
                  <td className="py-2">{item.model_used}</td>
                  <td className="py-2">{item.inference_time_ms.toFixed(2)}</td>
                  <td className="py-2">{item.resource_usage.compute_cost_units}x</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default App;
