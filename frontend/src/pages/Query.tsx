import { useState } from "react";
import { queryDocuments } from "../api/client";
import type { QueryResponse, Claim } from "../api/client";
import { Search, CheckCircle, XCircle, AlertCircle, Cpu, FileText, ChevronDown, ChevronUp, Loader2 } from "lucide-react";

export function Query() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [expandedFeatures, setExpandedFeatures] = useState<Record<string, boolean>>({});

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const res = await queryDocuments({
        question: question.trim(),
        top_k: topK,
      });
      setResponse(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "Failed to execute query.");
    } finally {
      setLoading(false);
    }
  };

  const toggleFeatures = (claimId: string) => {
    setExpandedFeatures((prev) => ({
      ...prev,
      [claimId]: !prev[claimId],
    }));
  };

  const renderStatusBadge = (status: string) => {
    switch (status) {
      case "SUPPORT":
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-green-100 text-green-800 border border-green-200">
            <CheckCircle className="w-3.5 h-3.5 mr-1" />
            SUPPORT
          </span>
        );
      case "CONTRADICT":
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200">
            <XCircle className="w-3.5 h-3.5 mr-1" />
            CONTRADICT
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-200">
            <AlertCircle className="w-3.5 h-3.5 mr-1" />
            UNVERIFIED
          </span>
        );
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Grounded RAG & Claim Verification</h1>
        <p className="text-sm text-gray-600 mt-1">
          Ask questions across indexed research documents. ResearchGuard AI extracts atomic claims and verifies them using our Phase 5 trained Logistic Regression classifier.
        </p>
      </div>

      {/* Query Form Card */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <form onSubmit={handleQuery} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Research Question
            </label>
            <div className="relative">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. How does cell division relate to cyclin regulation?"
                className="w-full px-4 py-2.5 pl-10 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-gray-900"
              />
              <Search className="w-5 h-5 text-gray-400 absolute left-3 top-3" />
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-600">Retrieval Depth (top-k):</label>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="border border-gray-300 rounded px-2 py-1 text-sm bg-white"
              >
                <option value={3}>3 chunks</option>
                <option value={5}>5 chunks</option>
                <option value={10}>10 chunks</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={loading || !question.trim()}
              className="inline-flex items-center px-5 py-2.5 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors shadow-sm"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                "Query & Verify"
              )}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Response View */}
      {response && (
        <div className="space-y-6">
          {/* Answer Card */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <div className="flex items-center justify-between border-b pb-3 mb-4">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <FileText className="w-5 h-5 text-blue-600 mr-2" />
                Grounded Answer
              </h2>
              <span className="text-xs text-gray-500 bg-gray-100 px-2.5 py-1 rounded-full">
                Response Time: {response.response_time_ms} ms
              </span>
            </div>
            <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">{response.answer}</p>
          </div>

          {/* Extracted Claims & Verification Section */}
          {response.claims && response.claims.length > 0 && (
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Cpu className="w-5 h-5 text-purple-600 mr-2" />
                Extracted Claims & ML Verification ({response.claims.length})
              </h2>

              <div className="space-y-4">
                {response.claims.map((claim: Claim, idx: number) => {
                  const isExpanded = !!expandedFeatures[claim.claim_id];
                  const hasFeatures = claim.tfidf_similarity !== undefined && claim.tfidf_similarity !== null;

                  return (
                    <div
                      key={claim.claim_id || idx}
                      className="border border-gray-200 rounded-lg p-4 bg-gray-50/50 hover:bg-white transition-colors"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <span className="text-xs font-semibold text-gray-500">
                              Claim #{claim.claim_index + 1}
                            </span>
                            {renderStatusBadge(claim.verification_status)}
                            {claim.verification_method && (
                              <span className="text-xs font-mono text-gray-500 bg-gray-100 px-2 py-0.5 rounded border border-gray-200">
                                {claim.verification_method}
                              </span>
                            )}
                          </div>
                          <p className="text-sm font-medium text-gray-900 mt-1">{claim.claim_text}</p>
                        </div>

                        {claim.confidence_score !== undefined && claim.confidence_score !== null && (
                          <div className="text-right flex-shrink-0">
                            <div className="text-xs text-gray-500">Model Confidence</div>
                            <div className="text-base font-bold text-gray-900">
                              {(claim.confidence_score * 100).toFixed(1)}%
                            </div>
                          </div>
                        )}
                      </div>

                      {/* Feature Breakdown Toggle */}
                      {hasFeatures && (
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <button
                            onClick={() => toggleFeatures(claim.claim_id)}
                            className="inline-flex items-center text-xs font-medium text-blue-600 hover:text-blue-800"
                          >
                            {isExpanded ? (
                              <>
                                <ChevronUp className="w-3.5 h-3.5 mr-1" /> Hide 5-Feature Vector Details
                              </>
                            ) : (
                              <>
                                <ChevronDown className="w-3.5 h-3.5 mr-1" /> View 5-Feature Vector Details
                              </>
                            )}
                          </button>

                          {isExpanded && (
                            <div className="mt-2 grid grid-cols-2 md:grid-cols-5 gap-2 bg-gray-100 p-3 rounded text-xs">
                              <div>
                                <span className="text-gray-500 block">Cosine Similarity</span>
                                <span className="font-mono font-semibold text-gray-800">
                                  {claim.cosine_similarity?.toFixed(4) ?? "N/A"}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-500 block">TF-IDF Similarity</span>
                                <span className="font-mono font-semibold text-gray-800">
                                  {claim.tfidf_similarity?.toFixed(4) ?? "N/A"}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-500 block">Keyword Overlap</span>
                                <span className="font-mono font-semibold text-gray-800">
                                  {claim.keyword_overlap !== undefined ? (claim.keyword_overlap * 100).toFixed(1) + "%" : "N/A"}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-500 block">Claim Words</span>
                                <span className="font-mono font-semibold text-gray-800">
                                  {claim.claim_length ?? "N/A"}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-500 block">Evidence Words</span>
                                <span className="font-mono font-semibold text-gray-800">
                                  {claim.evidence_length ?? "N/A"}
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Evidence List */}
                      {claim.evidence && claim.evidence.length > 0 && (
                        <div className="mt-3 space-y-2">
                          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                            Supporting Evidence ({claim.evidence.length})
                          </span>
                          {claim.evidence.map((ev, eIdx) => (
                            <div
                              key={eIdx}
                              className="text-xs bg-white border border-gray-200 p-2.5 rounded text-gray-700 space-y-1"
                            >
                              <div className="flex justify-between items-center text-gray-500">
                                <span>Page {ev.page_number} (Rank #{ev.rank})</span>
                                <span className="font-mono">Similarity: {ev.similarity_score.toFixed(4)}</span>
                              </div>
                              <p className="italic text-gray-800">"{ev.evidence_text}"</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Sources Section */}
          {response.sources && response.sources.length > 0 && (
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900 mb-3">Retrieved Document Sources</h2>
              <div className="space-y-3">
                {response.sources.map((src, idx) => (
                  <div key={idx} className="p-3 bg-gray-50 border border-gray-200 rounded text-xs space-y-1">
                    <div className="flex justify-between text-gray-500 font-medium">
                      <span>Chunk #{src.chunk_index} — Page {src.page_number}</span>
                      <span className="font-mono">Similarity: {src.similarity_score.toFixed(4)}</span>
                    </div>
                    <p className="text-gray-700">{src.content}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
