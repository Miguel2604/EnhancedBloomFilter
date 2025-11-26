import { FilterResult } from "@/types/simulation";
import { DashboardCard } from "./ui/dashboard-card";
import { ArrowUpRight, ArrowDownRight, Activity, Database, Zap, Clock } from 'lucide-react';
import { formatThroughput } from "@/lib/utils";

interface MetricsSummaryProps {
  results: FilterResult[];
}

export function MetricsSummary({ results }: MetricsSummaryProps) {
  if (!results.length) return null;

  // Find best performers
  const bestFpr = [...results].sort((a, b) => a.fpr - b.fpr)[0];
  const bestThroughput = [...results].sort((a, b) => (b.throughput || 0) - (a.throughput || 0))[0];
  const bestThroughputValue = bestThroughput?.throughput || 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {/* Hero Card - Lowest FPR */}
      <div className="bg-black text-white p-6 border border-black flex flex-col justify-between h-full">
        <div>
          <div className="flex items-center gap-2 mb-4 border-b border-white/20 pb-2">
            <Activity className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-widest">Lowest FPR</span>
          </div>
          <div className="text-4xl font-bold mb-1 tracking-tighter">{(bestFpr.fpr * 100).toFixed(4)}%</div>
          <div className="text-sm text-white/80 mt-2 font-mono">{bestFpr.label}</div>
        </div>
        <div className="mt-4 flex items-center text-xs text-black bg-white w-fit px-2 py-1 font-bold uppercase">
          <ArrowDownRight className="w-3 h-3 mr-1" />
          Optimal
        </div>
      </div>

      <DashboardCard className="bg-white">
        <div className="flex items-center gap-2 text-gray-500 mb-4 border-b border-black pb-2">
          <Zap className="w-4 h-4 text-black" />
          <span className="text-xs font-bold uppercase tracking-widest text-black">Peak Throughput</span>
        </div>
        <div className="text-3xl font-bold text-black mb-1 tracking-tighter">
          {formatThroughput(bestThroughputValue)}
        </div>
        <div className="text-xs text-gray-500 font-mono mt-1">ops/sec ({bestThroughput.label})</div>
        <div className="mt-4 flex items-center text-xs text-white bg-black w-fit px-2 py-1 font-bold uppercase">
          <ArrowUpRight className="w-3 h-3 mr-1" />
          High Perf
        </div>
      </DashboardCard>

      <DashboardCard className="bg-white">
        <div className="flex items-center gap-2 text-gray-500 mb-4 border-b border-black pb-2">
          <Database className="w-4 h-4 text-black" />
          <span className="text-xs font-bold uppercase tracking-widest text-black">Avg Memory</span>
        </div>
        <div className="text-3xl font-bold text-black mb-1 tracking-tighter">
          {((results.reduce((acc, r) => acc + (r.memoryBytes || 0), 0) / results.length) / 1024).toFixed(1)} KB
        </div>
        <div className="text-xs text-gray-500 font-mono mt-1">Per filter instance</div>
      </DashboardCard>

      <DashboardCard className="bg-white">
        <div className="flex items-center gap-2 text-gray-500 mb-4 border-b border-black pb-2">
          <Clock className="w-4 h-4 text-black" />
          <span className="text-xs font-bold uppercase tracking-widest text-black">Avg Creation Time</span>
        </div>
        <div className="text-3xl font-bold text-black mb-1 tracking-tighter">
          {(results.reduce((acc, r) => acc + (r.creationTimeMs || 0), 0) / results.length).toFixed(1)} ms
        </div>
        <div className="text-xs text-gray-500 font-mono mt-1">Processing latency</div>
      </DashboardCard>
    </div>
  );
}
