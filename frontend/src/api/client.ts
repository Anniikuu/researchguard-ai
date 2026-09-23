import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface HealthResponse {
  status: string;
  services: {
    database: string;
    ollama: string;
  };
}

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await api.get('/health');
  return response.data;
};

export default api;
