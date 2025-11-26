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
  const { results, isLoading, error, run, config, setConfig } = useSimulation();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className="min-h-screen bg-white font-sans text-black">
      
      <main className="transition-all duration-300">
        {/* Header */}
        <header className="h-24 px-8 flex items-center justify-between border-b border-black mb-8">
          <h1 className="text-3xl font-bold uppercase tracking-tighter">Learned Bloom Filter Simulation</h1>
        </header>

        <div className="px-8 pb-12">
          {/* Loading Overlay */}
          {isLoading && (
             <div className="fixed inset-0 z-50 flex items-center justify-center bg-white/80 backdrop-blur-none">
               <div className="flex flex-col items-center gap-6 p-12 bg-white border border-black animate-in zoom-in-95 duration-300">
                 <div className="relative w-16 h-16">
                   <div className="absolute inset-0 border-4 border-gray-200" />
                   <div className="absolute inset-0 border-4 border-black border-t-transparent animate-spin" />
                   <Activity className="absolute inset-0 m-auto w-6 h-6 text-black animate-pulse" />
                 </div>
                 <div className="text-center">
                   <h3 className="text-lg font-bold uppercase tracking-widest text-black">Processing</h3>
                   <p className="text-sm font-mono text-gray-500 mt-2">Training models...</p>
                 </div>
               </div>
             </div>
          )}

          {/* Top Metrics Row Removed */}

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Column: Config & Summary */}
            <div className="lg:col-span-4 space-y-8">
              <DashboardCard title="Configuration" className="h-auto">
                <SimulationConfigForm 
                  onSubmit={run} 
                  isLoading={isLoading} 
                  config={config}
                  onConfigChange={setConfig}
                />
              </DashboardCard>
            </div>

            {/* Right Column: Charts */}
            <div className="lg:col-span-8 space-y-8">
              {/* Main Chart Container */}
              <DashboardCard 
                title="Performance Metrics" 
                className="min-h-[400px]"
              >
                <div className="space-y-8">
                  {/* FPR Chart */}
                  <div>
                    <h3 className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-4">False Positive Rate History</h3>
                    {results ? (
                      <FprChart results={results} />
                    ) : (
                      <div className="h-[300px] flex items-center justify-center text-gray-400 font-mono border border-dashed border-gray-300">
                        NO SIMULATION DATA
                      </div>
                    )}
                  </div>

                  {/* Secondary Metrics Row */}
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 pt-8 border-t border-black">
                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-4">Throughput (ops/sec)</h3>
                      {results ? (
                        <MetricsComparison 
                          results={results} 
                          metric="throughput" 
                          formatter={(val) => formatThroughput(val, { millionDecimals: 2, thousandDecimals: 1 })}
                        />
                      ) : (
                        <div className="h-[200px] bg-gray-50 border border-gray-200" />
                      )}
                    </div>

                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-4">Memory Usage</h3>
                      {results ? (
                        <MetricsComparison 
                          results={results} 
                          metric="memoryBytes" 
                          unit=" KB"
                          formatter={(val) => (val / 1024).toFixed(2)}
                        />
                      ) : (
                        <div className="h-[200px] bg-gray-50 border border-gray-200" />
                      )}
                    </div>

                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-4">Average Retraining Time (ms)</h3>
                      {results ? (
                        <MetricsComparison 
                          results={results} 
                          metric="creationTimeMs" 
                          unit=" ms"
                          formatter={(val) => val.toFixed(1)}
                        />
                      ) : (
                        <div className="h-[200px] bg-gray-50 border border-gray-200" />
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
