"use client";

import { useComparisonContext } from "@/context/comparison-context";
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
} from "recharts";

// Helper to calculate standard deviation
function calculateStdDev(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
  const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(avgSquaredDiff);
}

export default function ComparisonPage() {
  const { data, isLoading, error, config, setConfig, runComparison } = useComparisonContext();

  const handleRunComparison = () => {
    runComparison(config);
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
    <div className="min-h-screen bg-white font-sans text-black">
      <main className="p-8 max-w-7xl mx-auto space-y-8">
        <header className="border-b border-black pb-6">
          <h1 className="text-4xl font-bold uppercase tracking-tighter text-black">
            Enhancement Comparison
          </h1>
          <p className="text-gray-500 mt-2 font-mono uppercase tracking-widest text-sm">
            Basic LBF vs Combined Enhanced LBF
          </p>
        </header>

        {/* Configuration Panel */}
        <DashboardCard title="Simulation Configuration">
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-xs font-bold uppercase tracking-widest text-black mb-2">
                  Dataset Size: {config.datasetSize.toLocaleString()} items
                </label>
                <Slider
                  value={[config.datasetSize]}
                  onValueChange={(val) => setConfig({ ...config, datasetSize: val[0] })}
                  min={1000}
                  max={30000}
                  step={1000}
                  className="w-full"
                />
              </div>
              <div>
                <label className="block text-xs font-bold uppercase tracking-widest text-black mb-2">
                  Query Count: {config.queryCount.toLocaleString()} queries
                </label>
                <Slider
                  value={[config.queryCount]}
                  onValueChange={(val) => setConfig({ ...config, queryCount: val[0] })}
                  min={1000}
                  max={10000}
                  step={500}
                  className="w-full"
                />
              </div>
            </div>
            <Button
              onClick={handleRunComparison}
              disabled={isLoading}
              className="w-full md:w-auto bg-black text-white rounded-none hover:bg-gray-900 border border-black uppercase font-bold"
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
              <p className="text-red-600 text-sm mt-2 font-bold uppercase">{error}</p>
            )}
          </div>
        </DashboardCard>

        {/* Loading State */}
        {isLoading && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/80">
            <div className="flex flex-col items-center gap-6 p-12 bg-white border border-black shadow-none">
              <div className="relative w-16 h-16">
                <div className="absolute inset-0 border-4 border-gray-200" />
                <div className="absolute inset-0 border-4 border-black border-t-transparent animate-spin" />
                <Activity className="absolute inset-0 m-auto w-6 h-6 text-black animate-pulse" />
              </div>
              <div className="text-center">
                <h3 className="text-lg font-bold uppercase tracking-widest text-black">Running Comparison</h3>
                <p className="text-sm font-mono text-gray-500 mt-2">Testing implementations...</p>
              </div>
            </div>
          </div>
        )}

        {data && (
          <>
            {/* Key Improvements Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white text-black border border-black p-6">
                <div className="flex items-center gap-2 text-gray-500 mb-2 pb-2 border-b border-black">
                  <Zap className="w-4 h-4 text-black" />
                  <span className="text-xs font-bold uppercase tracking-widest text-black">Query Speedup</span>
                </div>
                <div className="text-5xl font-black mb-1 tracking-tighter">
                  {data.improvements.query_speedup.toFixed(2)}x
                </div>
                <p className="text-xs font-mono uppercase text-gray-500">faster queries</p>
              </div>

              <div className="bg-black text-white border border-black p-6">
                <div className="flex items-center gap-2 text-white/60 mb-2 pb-2 border-b border-white">
                  <Clock className="w-4 h-4 text-white" />
                  <span className="text-xs font-bold uppercase tracking-widest text-white">Update Speedup</span>
                </div>
                <div className="text-5xl font-black mb-1 tracking-tighter">
                  {data.improvements.update_speedup.toFixed(0)}x
                </div>
                <p className="text-xs font-mono uppercase text-white/60">faster updates (O(1) vs O(n))</p>
              </div>

              <div className="bg-white text-black border border-black p-6">
                <div className="flex items-center gap-2 text-gray-500 mb-2 pb-2 border-b border-black">
                  <TrendingDown className="w-4 h-4 text-black" />
                  <span className="text-xs font-bold uppercase tracking-widest text-black">FPR Reduction</span>
                </div>
                <div className="text-5xl font-black mb-1 tracking-tighter">
                  {data.improvements.fpr_stability_improvement.toFixed(1)}x
                </div>
                <p className="text-xs font-mono uppercase text-gray-500">lower false positives</p>
              </div>
            </div>

            {/* Before/After Comparison Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Basic LBF - Problems */}
              <DashboardCard className="border border-red-600 bg-white">
                <div className="flex items-center gap-3 mb-6 pb-4 border-b border-red-600">
                  <div className="w-8 h-8 flex items-center justify-center border border-red-600 bg-red-600 text-white">
                    <AlertTriangle className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold uppercase tracking-tight text-red-600">
                      Before: Basic LBF
                    </h3>
                    <p className="text-xs font-bold uppercase tracking-widest text-red-400">
                      3 Critical Problems
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  {data.basic_lbf.problems.map((problem, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-3 p-3 bg-white border border-red-100"
                    >
                      <TrendingDown className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" />
                      <span className="text-sm font-mono text-gray-700">{problem}</span>
                    </div>
                  ))}
                </div>

                <div className="mt-6 pt-6 border-t border-red-200">
                  <h4 className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-3">
                    Performance Metrics
                  </h4>
                  <div className="grid grid-cols-2 gap-4 text-sm font-mono">
                    <div>
                      <span className="text-gray-500 block text-xs uppercase">Throughput</span>
                      <span className="font-bold">
                        {(data.basic_lbf.throughput / 1000).toFixed(1)}K ops/s
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500 block text-xs uppercase">Update</span>
                      <span className="font-bold">
                        {data.basic_lbf.update_time_ms.toFixed(1)}ms ({data.basic_lbf.update_complexity})
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500 block text-xs uppercase">False Positive Rate</span>
                      <span className="font-bold text-red-600">
                        {data.basic_lbf.fpr_mean.toFixed(2)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500 block text-xs uppercase">Memory</span>
                      <span className="font-bold">
                        {data.basic_lbf.memory_mb.toFixed(2)} MB
                      </span>
                    </div>
                  </div>
                </div>
              </DashboardCard>

              {/* Enhanced LBF - Solutions */}
              <DashboardCard className="border border-black bg-white">
                <div className="flex items-center gap-3 mb-6 pb-4 border-b border-black">
                  <div className="w-8 h-8 flex items-center justify-center border border-black bg-black text-white">
                    <CheckCircle2 className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold uppercase tracking-tight text-black">
                      After: Enhanced LBF
                    </h3>
                    <p className="text-xs font-bold uppercase tracking-widest text-gray-500">
                      3 Solutions Applied
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  {data.enhanced_lbf.solutions.map((solution, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-3 p-3 bg-white border border-gray-200"
                    >
                      <CheckCircle2 className="w-4 h-4 text-black mt-0.5 flex-shrink-0" />
                      <span className="text-sm font-mono text-black">{solution}</span>
                    </div>
                  ))}
                </div>

                <div className="mt-6 pt-6 border-t border-black">
                  <h4 className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-3">
                    Performance Metrics
                  </h4>
                  <div className="grid grid-cols-2 gap-4 text-sm font-mono">
                    <div>
                      <span className="text-gray-500 block text-xs uppercase">Throughput</span>
                      <span className="font-bold text-black">
                        {(data.enhanced_lbf.throughput / 1000).toFixed(1)}K ops/s
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500 block text-xs uppercase">Update</span>
                      <span className="font-bold text-black">
                        {data.enhanced_lbf.update_time_ms.toFixed(3)}ms ({data.enhanced_lbf.update_complexity})
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500 block text-xs uppercase">False Positive Rate</span>
                      <span className="font-bold text-black">
                        {data.enhanced_lbf.fpr_mean.toFixed(2)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500 block text-xs uppercase">Memory</span>
                      <span className="font-bold">
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
                  <div key={key} className="p-4 border border-gray-200 bg-white">
                    <h4 className="text-xs font-bold uppercase tracking-widest text-black mb-4 pb-2 border-b border-gray-100">
                      {key.replace(/_/g, " ")}
                    </h4>
                    <div className="space-y-3 font-mono text-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500 uppercase">Before</span>
                        <span className="text-red-600 font-bold text-right">
                          {value.before}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500 uppercase">After</span>
                        <span className="text-black font-bold text-right">
                          {value.after}
                        </span>
                      </div>
                      <div className="pt-2 border-t border-black mt-2 text-right">
                        <span className="text-xs font-bold uppercase text-black bg-gray-100 px-2 py-1">
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
              {/* FPR Comparison Chart */}
              <DashboardCard title="FPR Comparison">
                <p className="text-xs font-mono text-gray-500 mb-4 uppercase">
                  False positive rate measured across 20 rounds with varying query distributions
                </p>
                {/* FPR Standard Deviation Display */}
                <div className="grid grid-cols-2 gap-4 mb-6 p-4 bg-gray-50 border border-gray-200">
                  <div>
                    <span className="text-xs font-bold uppercase tracking-widest text-red-600 block mb-1">Basic LBF</span>
                    <div className="font-mono text-sm">
                      <span className="text-gray-500">Mean:</span> <span className="font-bold">{data.basic_lbf.fpr_mean.toFixed(3)}%</span>
                      <span className="mx-2 text-gray-300">|</span>
                      <span className="text-gray-500">Std Dev:</span> <span className="font-bold text-red-600">{calculateStdDev(data.basic_lbf.fpr_history).toFixed(4)}%</span>
                    </div>
                  </div>
                  <div>
                    <span className="text-xs font-bold uppercase tracking-widest text-black block mb-1">Enhanced LBF</span>
                    <div className="font-mono text-sm">
                      <span className="text-gray-500">Mean:</span> <span className="font-bold">{data.enhanced_lbf.fpr_mean.toFixed(3)}%</span>
                      <span className="mx-2 text-gray-300">|</span>
                      <span className="text-gray-500">Std Dev:</span> <span className="font-bold text-black">{calculateStdDev(data.enhanced_lbf.fpr_history).toFixed(4)}%</span>
                    </div>
                  </div>
                </div>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={fprChartData}>
                      <CartesianGrid vertical={false} stroke="#e5e5e5" />
                      <XAxis
                        dataKey="round"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#000", fontSize: 10, fontFamily: "sans-serif", fontWeight: "bold" }}
                      />
                      <YAxis
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#000", fontSize: 10, fontFamily: "sans-serif", fontWeight: "bold" }}
                        tickFormatter={(val) => `${val.toFixed(1)}%`}
                      />
                      <Tooltip
                        formatter={(value: number) => [`${value.toFixed(2)}%`, ""]}
                        contentStyle={{
                          borderRadius: "0px",
                          border: "1px solid black",
                          boxShadow: "none",
                          fontFamily: "sans-serif"
                        }}
                        itemStyle={{ fontWeight: "bold" }}
                      />
                      <Legend wrapperStyle={{ fontFamily: 'sans-serif', fontSize: '10px', fontWeight: 'bold' }} />
                      <Line
                        type="linear"
                        dataKey="Basic LBF"
                        stroke="#dc2626"
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        type="linear"
                        dataKey="Enhanced LBF"
                        stroke="#000000"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </DashboardCard>

              {/* Performance Bar Chart */}
              <DashboardCard title="Performance Metrics">
                <p className="text-xs font-mono text-gray-500 mb-6 uppercase">
                  Side-by-side comparison of key metrics
                </p>
                <div className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={metricsBarData} layout="vertical">
                      <CartesianGrid horizontal={true} vertical={false} stroke="#e5e5e5" />
                      <XAxis type="number" axisLine={false} tickLine={false} />
                      <YAxis
                        type="category"
                        dataKey="metric"
                        axisLine={false}
                        tickLine={false}
                        width={100}
                        tick={{ fill: "#000", fontSize: 10, fontFamily: "sans-serif", fontWeight: "bold" }}
                      />
                      <Tooltip
                        contentStyle={{
                          borderRadius: "0px",
                          border: "1px solid black",
                          boxShadow: "none",
                          fontFamily: "sans-serif"
                        }}
                        itemStyle={{ fontWeight: "bold" }}
                      />
                      <Legend wrapperStyle={{ fontFamily: 'sans-serif', fontSize: '10px', fontWeight: 'bold' }} />
                      <Bar dataKey="Basic" fill="#dc2626" radius={[0, 0, 0, 0]} />
                      <Bar dataKey="Enhanced" fill="#000000" radius={[0, 0, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </DashboardCard>
            </div>

            {/* Study Objectives Met */}
            <DashboardCard title="Study Objectives Achieved">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="p-6 border border-black bg-white">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-6 h-6 border border-black bg-black flex items-center justify-center">
                      <span className="text-white font-bold text-xs">1</span>
                    </div>
                    <h4 className="font-bold text-black uppercase text-sm">Cache Optimization</h4>
                  </div>
                  <p className="text-sm font-mono text-gray-600 mb-4">
                    64-byte aligned memory blocks reduce cache misses.
                  </p>
                  <div className="text-3xl font-black text-black tracking-tighter">
                    {data.improvements.query_speedup.toFixed(2)}x
                  </div>
                  <p className="text-xs font-bold uppercase text-gray-400">throughput improvement</p>
                </div>

                <div className="p-6 border border-black bg-white">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-6 h-6 border border-black bg-black flex items-center justify-center">
                      <span className="text-white font-bold text-xs">2</span>
                    </div>
                    <h4 className="font-bold text-black uppercase text-sm">Incremental Learning</h4>
                  </div>
                  <p className="text-sm font-mono text-gray-600 mb-4">
                    O(1) updates using Passive-Aggressive online learning.
                  </p>
                  <div className="text-3xl font-black text-black tracking-tighter">
                    {data.improvements.update_speedup.toFixed(0)}x
                  </div>
                  <p className="text-xs font-bold uppercase text-gray-400">faster updates</p>
                </div>

                <div className="p-6 border border-black bg-white">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-6 h-6 border border-black bg-black flex items-center justify-center">
                      <span className="text-white font-bold text-xs">3</span>
                    </div>
                    <h4 className="font-bold text-black uppercase text-sm">Adaptive Control</h4>
                  </div>
                  <p className="text-sm font-mono text-gray-600 mb-4">
                    PID controller + optimized feature extraction.
                  </p>
                  <div className="text-3xl font-black text-black tracking-tighter">
                    {data.improvements.fpr_stability_improvement.toFixed(1)}x
                  </div>
                  <p className="text-xs font-bold uppercase text-gray-400">lower FPR</p>
                </div>
              </div>
            </DashboardCard>
          </>
        )}

        {/* Initial State - No Data */}
        {!data && !isLoading && (
          <DashboardCard className="text-center py-24 bg-gray-50 border border-dashed border-gray-300">
            <div className="max-w-md mx-auto">
              <div className="w-16 h-16 border-2 border-black flex items-center justify-center mx-auto mb-6 bg-white">
                <Activity className="w-8 h-8 text-black" />
              </div>
              <h3 className="text-xl font-bold uppercase tracking-widest text-black mb-2">
                Ready to Compare
              </h3>
              <p className="text-gray-500 mb-8 font-mono text-sm">
                Initiate the comparison sequence to evaluate Basic vs Enhanced LBF performance.
              </p>
              <div className="grid grid-cols-3 gap-4 text-xs font-bold uppercase">
                <div className="p-3 border border-red-200 bg-white text-red-600">
                  <AlertTriangle className="w-4 h-4 mx-auto mb-2" />
                  <p>Cache Misses</p>
                </div>
                <div className="p-3 border border-red-200 bg-white text-red-600">
                  <AlertTriangle className="w-4 h-4 mx-auto mb-2" />
                  <p>Slow Retraining</p>
                </div>
                <div className="p-3 border border-red-200 bg-white text-red-600">
                  <AlertTriangle className="w-4 h-4 mx-auto mb-2" />
                  <p>Unstable FPR</p>
                </div>
              </div>
            </div>
          </DashboardCard>
        )}
      </main>
    </div>
  );
}
