export interface SimulationConfig {
  datasetSize: number;
  queryCount: number;
  selectedFilters: string[];
}

export interface FilterResult {
  id: string;
  label: string;
  fpr: number;
  fprVariance?: number;
  throughput?: number; // ops/sec
  memoryBytes?: number;
  creationTimeMs?: number;
  queryTimeMs?: number;
  fprHistory: number[]; // For line chart
}

export interface SimulationResponse {
  results: FilterResult[];
}

export const FILTER_OPTIONS = [
  { id: "standard_bf", label: "Standard Bloom Filter", color: "#8884d8" },
  { id: "basic_lbf", label: "Basic LBF", color: "#82ca9d" },
  { id: "cache_aligned_lbf", label: "Cache-Aligned LBF", color: "#ffc658" },
  { id: "incremental_lbf", label: "Incremental LBF", color: "#ff7300" },
  { id: "adaptive_lbf", label: "Adaptive LBF", color: "#0088fe" },
  { id: "combined_lbf", label: "Combined Enhanced LBF", color: "#00C49F" },
];
