"use client";

import { useState } from "react";
import { DashboardCard } from "@/components/ui/dashboard-card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { 
  Activity, 
  Zap, 
  Clock, 
  TrendingUp, 
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  ArrowRight,
  Play,
  RefreshCw
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
  Cell,
} from "recharts";

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

export default function ComparisonPage() {
  const [data, setData] = useState<ComparisonData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasetSize, setDatasetSize] = useState(10000);
  const [queryCount, setQueryCount] = useState(5000);

  const runComparison = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/compare-enhancements`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_size: datasetSize,
          query_count: queryCount,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to run comparison: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  const fprChartData = data
    ? data.basic_lbf.fpr_history.map((val, idx) => ({
        round: idx + 1,
        "Basic LBF": val,
        "Enhanced LBF": data.enhanced_lbf.fpr_history[idx] || 0,
      }))
    : [];

  const metricsBarData = data
    ? [
        {
          metric: "Throughput",
          Basic: data.basic_lbf.throughput / 1000,
          Enhanced: data.enhanced_lbf.throughput / 1000,
          unit: "K ops/s",
        },
        {
          metric: "Query Time",
          Basic: data.basic_lbf.avg_query_time_us,
          Enhanced: data.enhanced_lbf.avg_query_time_us,
          unit: "μs",
        },
        {
          metric: "Update Time",
          Basic: Math.min(data.basic_lbf.update_time_ms, 100),
          Enhanced: data.enhanced_lbf.update_time_ms,
          unit: "ms",
        },
      ]
    : [];

  return (
    <div className="min-h-screen bg-[#F8F9FC] font-sans text-gray-900">
      <main className="p-8 max-w-7xl mx-auto space-y-8">
        <header>
          <h1 className="text-3xl font-bold text-gray-900">
            Enhancement Comparison
          </h1>
          <p className="text-gray-500 mt-2">
            Compare Basic Learned Bloom Filter (with problems) vs Combined Enhanced LBF (with solutions)
          </p>
        </header>

        {/* Configuration Panel */}
        <DashboardCard title="Simulation Configuration">
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Dataset Size: {datasetSize.toLocaleString()} items
                </label>
                <Slider
                  value={[datasetSize]}
                  onValueChange={(val) => setDatasetSize(val[0])}
                  min={1000}
                  max={30000}
                  step={1000}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Query Count: {queryCount.toLocaleString()} queries
                </label>
                <Slider
                  value={[queryCount]}
                  onValueChange={(val) => setQueryCount(val[0])}
                  min={1000}
                  max={10000}
                  step={500}
                  className="w-full"
                />
              </div>
            </div>
            <Button
              onClick={runComparison}
              disabled={isLoading}
              className="w-full md:w-auto"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Running Comparison...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Run Comparison
                </>
              )}
            </Button>
            {error && (
              <p className="text-red-500 text-sm mt-2">{error}</p>
            )}
          </div>
        </DashboardCard>

        {/* Loading State */}
        {isLoading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/50 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-4 p-8 bg-white rounded-2xl shadow-2xl border border-gray-100">
              <div className="relative w-16 h-16">
                <div className="absolute inset-0 border-4 border-gray-100 rounded-full" />
                <div className="absolute inset-0 border-4 border-purple-600 border-t-transparent rounded-full animate-spin" />
                <Activity className="absolute inset-0 m-auto w-6 h-6 text-purple-600 animate-pulse" />
              </div>
              <div className="text-center">
                <h3 className="text-lg font-bold text-gray-900">Running Comparison</h3>
                <p className="text-sm text-gray-500">Testing both implementations...</p>
              </div>
            </div>
          </div>
        )}

        {data && (
          <>
            {/* Key Improvements Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-gradient-to-br from-green-500 to-green-600 text-white rounded-[2rem] p-6 shadow-lg">
                <div className="flex items-center gap-2 text-white/80 mb-2">
                  <Zap className="w-5 h-5" />
                  <span className="text-sm font-medium">Query Speedup</span>
                </div>
                <div className="text-4xl font-bold mb-1">
                  {data.improvements.query_speedup.toFixed(2)}x
                </div>
                <p className="text-sm text-white/80">faster queries</p>
              </div>

              <div className="bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-[2rem] p-6 shadow-lg">
                <div className="flex items-center gap-2 text-white/80 mb-2">
                  <Clock className="w-5 h-5" />
                  <span className="text-sm font-medium">Update Speedup</span>
                </div>
                <div className="text-4xl font-bold mb-1">
                  {data.improvements.update_speedup.toFixed(0)}x
                </div>
                <p className="text-sm text-white/80">faster updates (O(1) vs O(n))</p>
              </div>

              <div className="bg-gradient-to-br from-purple-500 to-purple-600 text-white rounded-[2rem] p-6 shadow-lg">
                <div className="flex items-center gap-2 text-white/80 mb-2">
                  <TrendingUp className="w-5 h-5" />
                  <span className="text-sm font-medium">FPR Stability</span>
                </div>
                <div className="text-4xl font-bold mb-1">
                  {data.improvements.fpr_stability_improvement.toFixed(1)}x
                </div>
                <p className="text-sm text-white/80">more stable</p>
              </div>
            </div>

            {/* Before/After Comparison Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Basic LBF - Problems */}
              <DashboardCard className="border-2 border-red-200 bg-red-50/30">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center">
                    <AlertTriangle className="w-5 h-5 text-red-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-gray-900">
                      Before: Basic Learned Bloom Filter
                    </h3>
                    <p className="text-sm text-red-600 font-medium">
                      3 Critical Problems
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  {data.basic_lbf.problems.map((problem, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-3 p-3 bg-white rounded-xl border border-red-100"
                    >
                      <TrendingDown className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-700">{problem}</span>
                    </div>
                  ))}
                </div>

                <div className="mt-6 pt-6 border-t border-red-200">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">
                    Performance Metrics
                  </h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Throughput:</span>
                      <span className="ml-2 font-medium">
                        {(data.basic_lbf.throughput / 1000).toFixed(1)}K ops/s
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Update:</span>
                      <span className="ml-2 font-medium">
                        {data.basic_lbf.update_time_ms.toFixed(1)}ms ({data.basic_lbf.update_complexity})
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">FPR Variance:</span>
                      <span className="ml-2 font-medium text-red-600">
                        ±{data.basic_lbf.fpr_variance_pct.toFixed(0)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Memory:</span>
                      <span className="ml-2 font-medium">
                        {data.basic_lbf.memory_mb.toFixed(2)} MB
                      </span>
                    </div>
                  </div>
                </div>
              </DashboardCard>

              {/* Enhanced LBF - Solutions */}
              <DashboardCard className="border-2 border-green-200 bg-green-50/30">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-full bg-green-100 flex items-center justify-center">
                    <CheckCircle2 className="w-5 h-5 text-green-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-gray-900">
                      After: Combined Enhanced LBF
                    </h3>
                    <p className="text-sm text-green-600 font-medium">
                      3 Solutions Applied
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  {data.enhanced_lbf.solutions.map((solution, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-3 p-3 bg-white rounded-xl border border-green-100"
                    >
                      <CheckCircle2 className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-700">{solution}</span>
                    </div>
                  ))}
                </div>

                <div className="mt-6 pt-6 border-t border-green-200">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">
                    Performance Metrics
                  </h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Throughput:</span>
                      <span className="ml-2 font-medium text-green-600">
                        {(data.enhanced_lbf.throughput / 1000).toFixed(1)}K ops/s
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Update:</span>
                      <span className="ml-2 font-medium text-green-600">
                        {data.enhanced_lbf.update_time_ms.toFixed(3)}ms ({data.enhanced_lbf.update_complexity})
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">FPR Variance:</span>
                      <span className="ml-2 font-medium text-green-600">
                        ±{data.enhanced_lbf.fpr_variance_pct.toFixed(0)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Memory:</span>
                      <span className="ml-2 font-medium">
                        {data.enhanced_lbf.memory_mb.toFixed(2)} MB
                      </span>
                    </div>
                  </div>
                </div>
              </DashboardCard>
            </div>

            {/* Detailed Improvement Breakdown */}
            <DashboardCard title="Improvement Details">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {Object.entries(data.improvements.summary).map(([key, value]) => (
                  <div key={key} className="p-4 bg-gray-50 rounded-xl">
                    <h4 className="text-sm font-semibold text-gray-900 mb-3 capitalize">
                      {key.replace(/_/g, " ")}
                    </h4>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500 w-16">Before:</span>
                        <span className="text-sm font-medium text-red-600">
                          {value.before}
                        </span>
                      </div>
                      <div className="flex items-center justify-center my-2">
                        <ArrowRight className="w-4 h-4 text-gray-400" />
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500 w-16">After:</span>
                        <span className="text-sm font-medium text-green-600">
                          {value.after}
                        </span>
                      </div>
                      <div className="pt-2 border-t border-gray-200 mt-2">
                        <span className="text-sm font-bold text-purple-600">
                          {value.improvement}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </DashboardCard>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* FPR Stability Chart */}
              <DashboardCard title="FPR Stability Comparison">
                <p className="text-sm text-gray-500 mb-4">
                  FPR measured across 20 rounds with varying query distributions
                </p>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={fprChartData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                      <XAxis
                        dataKey="round"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#9ca3af", fontSize: 12 }}
                      />
                      <YAxis
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#9ca3af", fontSize: 12 }}
                        tickFormatter={(val) => `${val.toFixed(1)}%`}
                      />
                      <Tooltip
                        formatter={(value: number) => [`${value.toFixed(2)}%`, ""]}
                        contentStyle={{
                          borderRadius: "12px",
                          border: "none",
                          boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                        }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="Basic LBF"
                        stroke="#ef4444"
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="Enhanced LBF"
                        stroke="#22c55e"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </DashboardCard>

              {/* Performance Bar Chart */}
              <DashboardCard title="Performance Metrics">
                <p className="text-sm text-gray-500 mb-4">
                  Side-by-side comparison of key metrics
                </p>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={metricsBarData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                      <XAxis type="number" axisLine={false} tickLine={false} />
                      <YAxis
                        type="category"
                        dataKey="metric"
                        axisLine={false}
                        tickLine={false}
                        width={100}
                      />
                      <Tooltip
                        contentStyle={{
                          borderRadius: "12px",
                          border: "none",
                          boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                        }}
                      />
                      <Legend />
                      <Bar dataKey="Basic" fill="#ef4444" radius={[0, 4, 4, 0]} />
                      <Bar dataKey="Enhanced" fill="#22c55e" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </DashboardCard>
            </div>

            {/* Study Objectives Met */}
            <DashboardCard title="Study Objectives Achieved">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="p-6 bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-8 h-8 rounded-lg bg-blue-500 flex items-center justify-center">
                      <span className="text-white font-bold text-sm">1</span>
                    </div>
                    <h4 className="font-semibold text-gray-900">Cache Optimization</h4>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">
                    64-byte aligned memory blocks reduce cache misses by organizing data for CPU cache lines.
                  </p>
                  <div className="text-2xl font-bold text-blue-600">
                    {data.improvements.query_speedup.toFixed(2)}x
                  </div>
                  <p className="text-xs text-gray-500">throughput improvement</p>
                </div>

                <div className="p-6 bg-gradient-to-br from-green-50 to-green-100 rounded-2xl">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-8 h-8 rounded-lg bg-green-500 flex items-center justify-center">
                      <span className="text-white font-bold text-sm">2</span>
                    </div>
                    <h4 className="font-semibold text-gray-900">Incremental Learning</h4>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">
                    O(1) updates using Passive-Aggressive online learning with reservoir sampling.
                  </p>
                  <div className="text-2xl font-bold text-green-600">
                    {data.improvements.update_speedup.toFixed(0)}x
                  </div>
                  <p className="text-xs text-gray-500">faster updates</p>
                </div>

                <div className="p-6 bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-8 h-8 rounded-lg bg-purple-500 flex items-center justify-center">
                      <span className="text-white font-bold text-sm">3</span>
                    </div>
                    <h4 className="font-semibold text-gray-900">Adaptive Control</h4>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">
                    PID controller dynamically adjusts threshold to maintain stable FPR under varying workloads.
                  </p>
                  <div className="text-2xl font-bold text-purple-600">
                    {data.improvements.fpr_stability_improvement.toFixed(1)}x
                  </div>
                  <p className="text-xs text-gray-500">more stable FPR</p>
                </div>
              </div>
            </DashboardCard>
          </>
        )}

        {/* Initial State - No Data */}
        {!data && !isLoading && (
          <DashboardCard className="text-center py-12">
            <div className="max-w-md mx-auto">
              <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-4">
                <Activity className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Ready to Compare
              </h3>
              <p className="text-gray-500 mb-6">
                Click &quot;Run Comparison&quot; to see how the Enhanced Learned Bloom Filter 
                solves the three critical problems of the basic implementation.
              </p>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div className="p-3 bg-red-50 rounded-xl">
                  <AlertTriangle className="w-5 h-5 text-red-500 mx-auto mb-2" />
                  <p className="text-gray-600">Poor Cache Locality</p>
                </div>
                <div className="p-3 bg-red-50 rounded-xl">
                  <AlertTriangle className="w-5 h-5 text-red-500 mx-auto mb-2" />
                  <p className="text-gray-600">O(n) Retraining</p>
                </div>
                <div className="p-3 bg-red-50 rounded-xl">
                  <AlertTriangle className="w-5 h-5 text-red-500 mx-auto mb-2" />
                  <p className="text-gray-600">Unstable FPR</p>
                </div>
              </div>
            </div>
          </DashboardCard>
        )}
      </main>
    </div>
  );
}
