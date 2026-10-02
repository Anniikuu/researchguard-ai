import axios from 'axios';

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data?.detail || error.message);
    return Promise.reject(error);
  }
);

export interface HealthResponse {
  status: string;
  database: string;
  ollama: string;
  embedding_model: string;
}

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
};

export interface SourceCitation {
  chunk_id: string;
  document_id: string;
  page_number: number;
  chunk_index: number;
  content: string;
  similarity_score: number;
  rank: number;
}

export interface QueryRequest {
  question: string;
  top_k?: number;
  document_id?: string;
}

export interface ClaimEvidence {
  chunk_id?: string;
  document_id?: string;
  page_number: number;
  evidence_text: string;
  similarity_score: number;
  rank: number;
}

export interface Claim {
  claim_id: string;
  claim_index: number;
  claim_text: string;
  verification_status: string;
  confidence_score?: number;
  verification_method?: string;
  cosine_similarity?: number;
  tfidf_similarity?: number;
  keyword_overlap?: number;
  claim_length?: number;
  evidence_length?: number;
  evidence: ClaimEvidence[];
}

export interface QueryResponse {
  question_id: string;
  answer_id: string;
  question: string;
  answer: string;
  sources: SourceCitation[];
  claims?: Claim[];
  response_time_ms: number;
  method: string;
}

export interface MetricDetail {
  mean: number;
  std: number;
}

export interface ModelMetrics {
  accuracy: MetricDetail;
  precision: MetricDetail;
  recall: MetricDetail;
  f1: MetricDetail;
  roc_auc: MetricDetail;
}

export interface SciFactFold {
  fold: number;
  train_size: number;
  val_size: number;
  baseline: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
  };
  logistic_regression: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc: number;
  };
}

export interface SciFactExperimentResults {
  experiment: string;
  research_question: string;
  dataset_metadata: {
    total_instances: number;
    support_count: number;
    contradict_count: number;
    support_pct: number;
    contradict_pct: number;
    unique_claims: number;
    unique_docs: number;
  };
  features: string[];
  cv_folds: SciFactFold[];
  aggregate_results: {
    baseline_cosine_threshold: ModelMetrics;
    proposed_logistic_regression: ModelMetrics;
  };
}

export const queryDocuments = async (data: QueryRequest): Promise<QueryResponse> => {
  const response = await apiClient.post<QueryResponse>('/query', data);
  return response.data;
};

export const getSciFactExperimentResults = async (): Promise<SciFactExperimentResults> => {
  const response = await apiClient.get<SciFactExperimentResults>('/experiments/scifact');
  return response.data;
};

export default apiClient;

