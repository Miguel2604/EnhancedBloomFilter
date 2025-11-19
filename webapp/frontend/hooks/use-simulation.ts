import { useState } from "react";
import { SimulationConfig, FilterResult } from "@/types/simulation";
import { runSimulation } from "@/lib/api";

export function useSimulation() {
  const [results, setResults] = useState<FilterResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (config: SimulationConfig) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await runSimulation(config);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  return { results, isLoading, error, run };
}
