import { useEffect, useState } from "react";
import { getSciFactExperimentResults } from "../api/client";
import type { SciFactExperimentResults } from "../api/client";
import { BarChart2, CheckCircle2, Info, Layers, Database, ShieldAlert } from "lucide-react";

export function Experiments() {
  const [data, setData] = useState<SciFactExperimentResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadResults() {
      try {
        const res = await getSciFactExperimentResults();
        setData(res);
      } catch (err: any) {
        setError(err.response?.data?.detail || err.message || "Failed to load experiment results.");
      } finally {
        setLoading(false);
      }
    }
    loadResults();
  }, []);

  if (loading) {
    return (
      <div className="p-8 text-center text-gray-500">
        <p>Loading Phase 5 SciFact Controlled Experiment Results...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
        {error || "No experiment results found."}
      </div>
    );
  }

  const baseline = data.aggregate_results.baseline_cosine_threshold;
  const lr = data.aggregate_results.proposed_logistic_regression;

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-12">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Phase 5 SciFact Controlled Experiment Dashboard</h1>
        <p className="text-sm text-gray-600 mt-1">
          Neutral benchmark comparison between the Cosine Similarity Baseline and the Supervised Logistic Regression classifier on 1,295 leakage-safe SciFact claim-evidence pairs.
        </p>
      </div>

      {/* Dataset Metadata Header Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center text-gray-500 text-xs font-medium mb-1">
            <Database className="w-4 h-4 mr-1 text-blue-500" /> Total Instances
          </div>
          <div className="text-2xl font-bold text-gray-900">{data.dataset_metadata.total_instances}</div>
          <div className="text-xs text-gray-500 mt-1">SUPPORT + CONTRADICT</div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center text-gray-500 text-xs font-medium mb-1">
            <CheckCircle2 className="w-4 h-4 mr-1 text-green-500" /> SUPPORT Class
          </div>
          <div className="text-2xl font-bold text-gray-900">{data.dataset_metadata.support_count}</div>
          <div className="text-xs text-gray-500 mt-1">{data.dataset_metadata.support_pct}% of dataset</div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center text-gray-500 text-xs font-medium mb-1">
            <ShieldAlert className="w-4 h-4 mr-1 text-red-500" /> CONTRADICT Class
          </div>
          <div className="text-2xl font-bold text-gray-900">{data.dataset_metadata.contradict_count}</div>
          <div className="text-xs text-gray-500 mt-1">{data.dataset_metadata.contradict_pct}% of dataset</div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center text-gray-500 text-xs font-medium mb-1">
            <Layers className="w-4 h-4 mr-1 text-purple-500" /> Leakage-Safe Splits
          </div>
          <div className="text-2xl font-bold text-gray-900">5-Fold CV</div>
          <div className="text-xs text-gray-500 mt-1">{data.dataset_metadata.unique_docs} unique documents</div>
        </div>
      </div>

      {/* Model Performance Comparison Table Card */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 space-y-4">
        <div className="flex items-center justify-between border-b pb-3">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center">
            <BarChart2 className="w-5 h-5 text-blue-600 mr-2" />
            Aggregate Model Performance Metrics (5-Fold Mean ± Std)
          </h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-gray-700">
            <thead className="bg-gray-50 text-xs uppercase text-gray-500 font-semibold border-b">
              <tr>
                <th className="py-3 px-4">Metric</th>
                <th className="py-3 px-4 bg-gray-100/50">Cosine Similarity Baseline</th>
                <th className="py-3 px-4 bg-blue-50/50">Supervised Logistic Regression</th>
                <th className="py-3 px-4">Comparison Finding</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              <tr>
                <td className="py-3.5 px-4 font-semibold text-gray-900">Precision</td>
                <td className="py-3.5 px-4 bg-gray-50/30 font-mono">
                  {baseline.precision.mean.toFixed(4)} ± {baseline.precision.std.toFixed(4)}
                </td>
                <td className="py-3.5 px-4 bg-blue-50/30 font-mono font-bold text-blue-700">
                  {lr.precision.mean.toFixed(4)} ± {lr.precision.std.toFixed(4)}
                </td>
                <td className="py-3.5 px-4 text-xs text-green-700 font-medium">
                  Logistic Regression achieved higher mean Precision (+0.0424)
                </td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-semibold text-gray-900">Accuracy</td>
                <td className="py-3.5 px-4 bg-gray-50/30 font-mono font-bold text-gray-900">
                  {baseline.accuracy.mean.toFixed(4)} ± {baseline.accuracy.std.toFixed(4)}
                </td>
                <td className="py-3.5 px-4 bg-blue-50/30 font-mono text-gray-700">
                  {lr.accuracy.mean.toFixed(4)} ± {lr.accuracy.std.toFixed(4)}
                </td>
                <td className="py-3.5 px-4 text-xs text-gray-600">
                  Cosine baseline achieved higher mean Accuracy
                </td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-semibold text-gray-900">Recall</td>
                <td className="py-3.5 px-4 bg-gray-50/30 font-mono font-bold text-gray-900">
                  {baseline.recall.mean.toFixed(4)} ± {baseline.recall.std.toFixed(4)}
                </td>
                <td className="py-3.5 px-4 bg-blue-50/30 font-mono text-gray-700">
                  {lr.recall.mean.toFixed(4)} ± {lr.recall.std.toFixed(4)}
                </td>
                <td className="py-3.5 px-4 text-xs text-gray-600">
                  Cosine baseline achieved higher mean Recall
                </td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-semibold text-gray-900">F1 Score</td>
                <td className="py-3.5 px-4 bg-gray-50/30 font-mono font-bold text-gray-900">
                  {baseline.f1.mean.toFixed(4)} ± {baseline.f1.std.toFixed(4)}
                </td>
                <td className="py-3.5 px-4 bg-blue-50/30 font-mono text-gray-700">
                  {lr.f1.mean.toFixed(4)} ± {lr.f1.std.toFixed(4)}
                </td>
                <td className="py-3.5 px-4 text-xs text-gray-600">
                  Cosine baseline achieved higher mean F1
                </td>
              </tr>
              <tr>
                <td className="py-3.5 px-4 font-semibold text-gray-900">ROC-AUC</td>
                <td className="py-3.5 px-4 bg-gray-50/30 font-mono">
                  {baseline.roc_auc.mean.toFixed(4)} ± {baseline.roc_auc.std.toFixed(4)}
                </td>
                <td className="py-3.5 px-4 bg-blue-50/30 font-mono">
                  {lr.roc_auc.mean.toFixed(4)} ± {lr.roc_auc.std.toFixed(4)}
                </td>
                <td className="py-3.5 px-4 text-xs text-gray-600">
                  Approximately comparable performance
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Neutral Scientific Interpretation Note */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 flex items-start space-x-3">
        <Info className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-amber-800 leading-relaxed">
          <strong className="font-semibold block mb-0.5">Scientific Interpretation:</strong>
          The Phase 5 controlled evaluation on SciFact yields a nuanced result: Supervised Logistic Regression achieved higher mean Precision than the cosine baseline (0.6892 vs 0.6468), reducing false positive support classifications. Conversely, the Cosine Baseline achieved higher mean Accuracy, Recall, and F1 due to high recall on the majority SUPPORT class. ROC-AUC is comparable between both methods.
        </div>
      </div>

      {/* Fold-by-Fold Breakdown Card */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-md font-semibold text-gray-900 mb-3">5-Fold Cross-Validation Breakdown</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left text-gray-600 border border-gray-200">
            <thead className="bg-gray-100 font-semibold uppercase text-gray-500">
              <tr>
                <th className="py-2.5 px-3 border-b">Fold #</th>
                <th className="py-2.5 px-3 border-b">Train / Val Size</th>
                <th className="py-2.5 px-3 border-b">Baseline Precision</th>
                <th className="py-2.5 px-3 border-b">LR Precision</th>
                <th className="py-2.5 px-3 border-b">Baseline F1</th>
                <th className="py-2.5 px-3 border-b">LR F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 font-mono">
              {data.cv_folds.map((f) => (
                <tr key={f.fold} className="hover:bg-gray-50">
                  <td className="py-2 px-3 font-semibold text-gray-900">Fold {f.fold}</td>
                  <td className="py-2 px-3">{f.train_size} / {f.val_size}</td>
                  <td className="py-2 px-3">{f.baseline.precision.toFixed(4)}</td>
                  <td className="py-2 px-3 text-blue-700 font-semibold">{f.logistic_regression.precision.toFixed(4)}</td>
                  <td className="py-2 px-3 font-semibold text-gray-900">{f.baseline.f1.toFixed(4)}</td>
                  <td className="py-2 px-3">{f.logistic_regression.f1.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
