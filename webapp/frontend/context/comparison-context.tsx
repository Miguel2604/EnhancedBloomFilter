"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

interface ComparisonData {
  basic_lbf: {
    name: string;
    training_time_ms: number;
    avg_query_time_us: number;
    throughput: number;
    update_time_ms: number;
    update_complexity: string;
    fpr_mean: number;
    fpr_std: number;
    fpr_variance_pct: number;
    fpr_history: number[];
    memory_bytes: number;
    memory_mb: number;
    problems: string[];
  };
  enhanced_lbf: {
    name: string;
    training_time_ms: number;
    avg_query_time_us: number;
    throughput: number;
    update_time_ms: number;
    update_complexity: string;
    fpr_mean: number;
    fpr_std: number;
    fpr_variance_pct: number;
    fpr_history: number[];
    memory_bytes: number;
    memory_mb: number;
    solutions: string[];
  };
  improvements: {
    throughput_increase_pct: number;
    query_speedup: number;
    update_speedup: number;
    fpr_stability_improvement: number;
    summary: {
      cache_locality: { before: string; after: string; improvement: string };
      update_complexity: { before: string; after: string; improvement: string };
      fpr_stability: { before: string; after: string; improvement: string };
    };
  };
}

interface ComparisonConfig {
  datasetSize: number;
  queryCount: number;
}

interface ComparisonContextValue {
  data: ComparisonData | null;
  isLoading: boolean;
  error: string | null;
  config: ComparisonConfig;
  setConfig: (config: ComparisonConfig) => void;
  runComparison: (config: ComparisonConfig) => Promise<void>;
}

const DEFAULT_CONFIG: ComparisonConfig = {
  datasetSize: 10000,
  queryCount: 5000,
};

const STORAGE_KEY = "comparison-state";

const ComparisonContext = createContext<ComparisonContextValue | undefined>(
  undefined
);

export function ComparisonProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [data, setData] = useState<ComparisonData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<ComparisonConfig>(DEFAULT_CONFIG);
  const hasHydrated = useRef(false);

  // Hydrate from session storage on mount
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const stored = window.sessionStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed.data) setData(parsed.data);
        if (parsed.config) setConfig(parsed.config);
      }
    } catch (err) {
      console.warn("Failed to hydrate comparison state", err);
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
        JSON.stringify({ data, config })
      );
    } catch (err) {
      console.warn("Failed to persist comparison state", err);
    }
  }, [data, config]);

  const runComparison = useCallback(async (newConfig: ComparisonConfig) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/compare-enhancements`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_size: newConfig.datasetSize,
          query_count: newConfig.queryCount,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to run comparison: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
      setConfig(newConfig);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const value: ComparisonContextValue = {
    data,
    isLoading,
    error,
    config,
    setConfig,
    runComparison,
  };

  return (
    <ComparisonContext.Provider value={value}>
      {children}
    </ComparisonContext.Provider>
  );
}

export function useComparisonContext() {
  const context = useContext(ComparisonContext);
  if (!context) {
    throw new Error("useComparison must be used within a ComparisonProvider");
  }
  return context;
}
