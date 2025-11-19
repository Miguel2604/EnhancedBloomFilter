import { FilterResult } from "@/types/simulation";
import { DashboardCard } from "./ui/dashboard-card";
import { ArrowUpRight, ArrowDownRight, Activity, Database, Zap, Clock } from 'lucide-react';

interface MetricsSummaryProps {
  results: FilterResult[];
}

export function MetricsSummary({ results }: MetricsSummaryProps) {
  if (!results.length) return null;

  // Find best performers
  const bestFpr = [...results].sort((a, b) => a.fpr - b.fpr)[0];
  const bestThroughput = [...results].sort((a, b) => (b.throughput || 0) - (a.throughput || 0))[0];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <div className="bg-black text-white rounded-[2rem] p-6 shadow-lg flex flex-col justify-between relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full -mr-10 -mt-10" />
        <div>
          <div className="flex items-center gap-2 text-white/60 mb-2">
            <Activity className="w-4 h-4" />
            <span className="text-sm font-medium">Lowest FPR</span>
          </div>
          <div className="text-3xl font-bold mb-1">{(bestFpr.fpr * 100).toFixed(4)}%</div>
          <div className="text-sm text-white/80">Winner: {bestFpr.label}</div>
        </div>
        <div className="mt-4 flex items-center text-xs text-green-400 bg-white/10 w-fit px-2 py-1 rounded-full">
          <ArrowDownRight className="w-3 h-3 mr-1" />
          Optimal Accuracy
        </div>
      </div>

      <DashboardCard className="bg-white">
        <div className="flex items-center gap-2 text-gray-500 mb-2">
          <Zap className="w-4 h-4" />
          <span className="text-sm font-medium">Peak Throughput</span>
        </div>
        <div className="text-3xl font-bold text-gray-900 mb-1">
          {((bestThroughput.throughput || 0) / 1000000).toFixed(2)}M
        </div>
        <div className="text-sm text-gray-500">ops/sec ({bestThroughput.label})</div>
        <div className="mt-4 flex items-center text-xs text-green-600 bg-green-50 w-fit px-2 py-1 rounded-full">
          <ArrowUpRight className="w-3 h-3 mr-1" />
          High Performance
        </div>
      </DashboardCard>

      <DashboardCard className="bg-white">
        <div className="flex items-center gap-2 text-gray-500 mb-2">
          <Database className="w-4 h-4" />
          <span className="text-sm font-medium">Avg Memory</span>
        </div>
        <div className="text-3xl font-bold text-gray-900 mb-1">
          {((results.reduce((acc, r) => acc + (r.memoryBytes || 0), 0) / results.length) / 1024).toFixed(1)} KB
        </div>
        <div className="text-sm text-gray-500">Per filter instance</div>
      </DashboardCard>

      <DashboardCard className="bg-white">
        <div className="flex items-center gap-2 text-gray-500 mb-2">
          <Clock className="w-4 h-4" />
          <span className="text-sm font-medium">Avg Creation Time</span>
        </div>
        <div className="text-3xl font-bold text-gray-900 mb-1">
          {(results.reduce((acc, r) => acc + (r.creationTimeMs || 0), 0) / results.length).toFixed(1)} ms
        </div>
        <div className="text-sm text-gray-500">Processing latency</div>
      </DashboardCard>
    </div>
  );
}
