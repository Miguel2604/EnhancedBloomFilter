"use client";

import { UrlProcessingVisualization } from "@/components/url-processing-visualization";

export default function RealtimePage() {
  return (
    <div className="min-h-screen bg-[#F8F9FC] font-sans text-gray-900">
      <main className="p-8 max-w-7xl mx-auto space-y-8">
        <header>
          <h1 className="text-3xl font-bold text-gray-900">Real-time Processing</h1>
          <p className="text-gray-500 mt-2">
            Visualize how the Enhanced Learned Bloom Filter processes URLs step-by-step through the ML model, cache, and backup filters.
          </p>
        </header>

        <UrlProcessingVisualization />
      </main>
    </div>
  );
}
