import { SimulationConfig, FilterResult, FILTER_OPTIONS } from "@/types/simulation";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

// Map frontend IDs to backend filter names
const ID_TO_NAME: Record<string, string> = {
  "standard_bf": "Standard BF",
  "basic_lbf": "Basic LBF",
  "cache_aligned_lbf": "Cache-Aligned LBF",
  "incremental_lbf": "Incremental LBF",
  "adaptive_lbf": "Adaptive LBF",
  "combined_lbf": "Combined Enhanced LBF",
};

// Map backend names to frontend IDs (reverse lookup or direct mapping)
const NAME_TO_ID_LABEL: Record<string, { id: string, label: string }> = {
  "Standard BF": { id: "standard_bf", label: "Standard Bloom Filter" },
  "Basic LBF": { id: "basic_lbf", label: "Basic LBF" },
  "Cache-Aligned LBF": { id: "cache_aligned_lbf", label: "Cache-Aligned LBF" },
  "Incremental LBF": { id: "incremental_lbf", label: "Incremental LBF" },
  "Adaptive LBF": { id: "adaptive_lbf", label: "Adaptive LBF" },
  "Combined Enhanced LBF": { id: "combined_lbf", label: "Combined Enhanced LBF" },
};

export async function runSimulation(config: SimulationConfig): Promise<FilterResult[]> {
  try {
    const backendFilters = config.selectedFilters
      .map(id => ID_TO_NAME[id])
      .filter(Boolean); // Filter out undefined mappings

    const response = await fetch(`${API_BASE_URL}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_size: config.datasetSize,
        query_count: config.queryCount,
        selected_filters: backendFilters.length > 0 ? backendFilters : ["Standard BF"],
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Simulation failed: ${response.status} ${errorText}`);
    }

    const data = await response.json();
    
    // Transform backend response to FilterResult
    return data.map((item: any) => {
      const mapping = NAME_TO_ID_LABEL[item.name] || { id: item.name.toLowerCase().replace(/ /g, "_"), label: item.name };
      
      return {
        id: mapping.id,
        label: mapping.label,
        fpr: item.fpr / 100, // Backend returns percentage (0-100), frontend expects fraction (0-1)
        fprVariance: item.fpr_variance,
        throughput: item.throughput,
        memoryBytes: item.memory_mb * 1024 * 1024, // MB to Bytes
        creationTimeMs: item.creation_time_ms,
        queryTimeMs: item.update_latency_ms, // Using update latency as a proxy or 0 if not exact match
        fprHistory: item.fpr_history.map((val: number) => val / 100), // Convert percentage to fraction
      };
    });

  } catch (error) {
    console.error("Simulation error:", error);
    throw error;
  }
}
