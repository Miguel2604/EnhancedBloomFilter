"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { runSimulation as fetchSimulation } from "@/lib/api";
import { FilterResult, SimulationConfig } from "@/types/simulation";

interface SimulationContextValue {
  results: FilterResult[] | null;
  isLoading: boolean;
  error: string | null;
  config: SimulationConfig;
  setConfig: (config: SimulationConfig) => void;
  run: (config: SimulationConfig) => Promise<void>;
}

const DEFAULT_CONFIG: SimulationConfig = {
  datasetSize: 100_000,
  queryCount: 50_000,
  selectedFilters: [
    "standard_bf",
    "basic_lbf",
    "cache_aligned_lbf",
    "combined_lbf",
  ],
};

const STORAGE_KEY = "simulation-state";

const SimulationContext = createContext<SimulationContextValue | undefined>(
  undefined
);

export function SimulationProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [results, setResults] = useState<FilterResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<SimulationConfig>(DEFAULT_CONFIG);
  const hasHydrated = useRef(false);

  // Hydrate from session storage on mount
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const stored = window.sessionStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed.results) setResults(parsed.results);
        if (parsed.config) setConfig(parsed.config);
      }
    } catch (err) {
      console.warn("Failed to hydrate simulation state", err);
    } finally {
      hasHydrated.current = true;
    }
  }, []);

  // Persist to session storage whenever relevant state changes
  useEffect(() => {
    if (!hasHydrated.current || typeof window === "undefined") return;
    try {
      window.sessionStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ results, config })
      );
    } catch (err) {
      console.warn("Failed to persist simulation state", err);
    }
  }, [results, config]);

  const run = useCallback(async (config: SimulationConfig) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchSimulation(config);
      setResults(data);
      setConfig(config);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "An unknown error occurred"
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  const value: SimulationContextValue = {
    results,
    isLoading,
    error,
    config,
    setConfig,
    run,
  };

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  );
}

export function useSimulationContext() {
  const context = useContext(SimulationContext);
  if (!context) {
    throw new Error("useSimulation must be used within a SimulationProvider");
  }
  return context;
}
