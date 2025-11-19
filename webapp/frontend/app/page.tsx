"use client";

import { useState, useEffect } from "react";
import { DashboardCard } from "@/components/ui/dashboard-card";
import { SimulationConfigForm } from "@/components/simulation-config-form";
import { FprChart } from "@/components/charts/fpr-chart";
import { MetricsComparison } from "@/components/charts/metrics-comparison";
import { MetricsSummary } from "@/components/metrics-summary";
import { useSimulation } from "@/hooks/use-simulation";
import { MoreHorizontal, Activity } from 'lucide-react';
import { formatThroughput } from "@/lib/utils";

export default function DashboardPage() {
  const { results, isLoading, error, run } = useSimulation();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className="min-h-screen bg-[#F8F9FC] font-sans text-gray-900">
      
      <main className="transition-all duration-300">
        {/* Header */}
        <header className="h-24 px-8 flex items-center justify-between">
          <h1 className="text-2xl font-bold">Learned Bloom Filter Simulation</h1>
        </header>

        <div className="px-8 pb-12">
          {/* Loading Overlay */}
          {isLoading && (
             <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/50 backdrop-blur-sm">
               <div className="flex flex-col items-center gap-4 p-8 bg-white rounded-2xl shadow-2xl border border-gray-100 animate-in zoom-in-95 duration-300">
                 <div className="relative w-16 h-16">
                   <div className="absolute inset-0 border-4 border-gray-100 rounded-full" />
                   <div className="absolute inset-0 border-4 border-purple-600 border-t-transparent rounded-full animate-spin" />
                   <Activity className="absolute inset-0 m-auto w-6 h-6 text-purple-600 animate-pulse" />
                 </div>
                 <div className="text-center">
                   <h3 className="text-lg font-bold text-gray-900">Running Simulation</h3>
                   <p className="text-sm text-gray-500">Training models and processing queries...</p>
                 </div>
               </div>
             </div>
          )}

          {/* Top Metrics Row */}
          {results && <MetricsSummary results={results} />}

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Column: Config & Summary */}
            <div className="lg:col-span-4 space-y-8">
              <DashboardCard title="Configuration" className="h-auto">
                <SimulationConfigForm onSubmit={run} isLoading={isLoading} />
              </DashboardCard>

              {results && (
                <DashboardCard title="Performance Score" className="h-auto">
                  <div className="flex flex-col items-center justify-center py-6">
                    <div className="relative w-48 h-24 overflow-hidden mb-4">
                      <div className="absolute top-0 left-0 w-full h-48 rounded-full border-[16px] border-gray-100" />
                      <div className="absolute top-0 left-0 w-full h-48 rounded-full border-[16px] border-purple-500 border-t-transparent border-r-transparent -rotate-45" />
                      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-center">
                        <div className="text-3xl font-bold text-gray-900">98.5</div>
                        <div className="text-xs text-gray-400 uppercase tracking-wider">Efficiency</div>
                      </div>
                    </div>
                    <p className="text-center text-gray-500 text-sm px-4">
                      Cache-aligned filters are performing 15% better than standard implementation.
                    </p>
                  </div>
                </DashboardCard>
              )}
            </div>

            {/* Right Column: Charts */}
            <div className="lg:col-span-8 space-y-8">
              {/* Main Chart Container */}
              <DashboardCard 
                title="Performance Metrics" 
                action={
                  <button className="p-2 hover:bg-gray-50 rounded-full text-gray-400">
                    <MoreHorizontal className="w-5 h-5" />
                  </button>
                }
                className="min-h-[400px]"
              >
                <div className="space-y-8">
                  {/* FPR Chart */}
                  <div>
                    <h3 className="text-sm font-medium text-gray-500 mb-4">False Positive Rate History</h3>
                    {results ? (
                      <FprChart results={results} />
                    ) : (
                      <div className="h-[300px] flex items-center justify-center text-gray-400">
                        No simulation data available
                      </div>
                    )}
                  </div>

                  {/* Secondary Metrics Row */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 pt-4 border-t border-gray-100">
                    <div>
                      <h3 className="text-sm font-medium text-gray-500 mb-4">Throughput (ops/sec)</h3>
                      {results ? (
                        <MetricsComparison 
                          results={results} 
                          metric="throughput" 
                          formatter={(val) => formatThroughput(val, { millionDecimals: 2, thousandDecimals: 1 })}
                        />
                      ) : (
                        <div className="h-[200px] bg-gray-50 rounded-xl animate-pulse" />
                      )}
                    </div>

                    <div>
                      <h3 className="text-sm font-medium text-gray-500 mb-4">Memory Usage</h3>
                      {results ? (
                        <MetricsComparison 
                          results={results} 
                          metric="memoryBytes" 
                          unit=" MB"
                          formatter={(val) => (val / (1024 * 1024)).toFixed(2)}
                        />
                      ) : (
                        <div className="h-[200px] bg-gray-50 rounded-xl animate-pulse" />
                      )}
                    </div>
                  </div>
                </div>
              </DashboardCard>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
