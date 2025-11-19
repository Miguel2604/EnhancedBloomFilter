import { useState } from "react";
import { SimulationConfig, FILTER_OPTIONS } from "@/types/simulation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Play, Settings2 } from 'lucide-react';

interface SimulationConfigFormProps {
  onSubmit: (config: SimulationConfig) => void;
  isLoading: boolean;
}

export function SimulationConfigForm({ onSubmit, isLoading }: SimulationConfigFormProps) {
  const [datasetSize, setDatasetSize] = useState(100000);
  const [queryCount, setQueryCount] = useState(50000);
  const [selectedFilters, setSelectedFilters] = useState<string[]>(["standard_bf", "basic_lbf", "cache_aligned_lbf", "combined_lbf"]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ datasetSize, queryCount, selectedFilters });
  };

  const toggleFilter = (id: string) => {
    setSelectedFilters((prev) =>
      prev.includes(id) ? prev.filter((f) => f !== id) : [...prev, id]
    );
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="dataset-size" className="text-gray-500 font-medium">Dataset Size</Label>
          <Input
            id="dataset-size"
            type="number"
            value={datasetSize}
            onChange={(e) => setDatasetSize(Number(e.target.value))}
            className="rounded-xl border-gray-200 bg-gray-50 focus:bg-white transition-colors"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="query-count" className="text-gray-500 font-medium">Query Count</Label>
          <Input
            id="query-count"
            type="number"
            value={queryCount}
            onChange={(e) => setQueryCount(Number(e.target.value))}
            className="rounded-xl border-gray-200 bg-gray-50 focus:bg-white transition-colors"
          />
        </div>
      </div>

      <div className="space-y-3">
        <Label className="text-gray-500 font-medium">Active Filters</Label>
        <div className="space-y-2">
          {FILTER_OPTIONS.map((filter) => (
            <div
              key={filter.id}
              className={`flex items-center space-x-3 p-3 rounded-xl transition-all cursor-pointer border ${
                selectedFilters.includes(filter.id)
                  ? "bg-purple-50 border-purple-200"
                  : "bg-gray-50 border-transparent hover:bg-gray-100"
              }`}
              onClick={() => toggleFilter(filter.id)}
            >
              <div
                className={`w-4 h-4 rounded-full flex items-center justify-center border ${
                  selectedFilters.includes(filter.id)
                    ? "border-purple-500 bg-purple-500"
                    : "border-gray-300"
                }`}
              >
                {selectedFilters.includes(filter.id) && (
                  <div className="w-1.5 h-1.5 bg-white rounded-full" />
                )}
              </div>
              <span className={`text-sm font-medium ${selectedFilters.includes(filter.id) ? "text-purple-900" : "text-gray-600"}`}>
                {filter.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      <Button 
        type="submit" 
        disabled={isLoading || selectedFilters.length === 0}
        className="w-full rounded-xl h-12 text-base font-medium bg-black hover:bg-gray-800 text-white shadow-lg shadow-gray-200"
      >
        {isLoading ? (
          <span className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Simulating...
          </span>
        ) : (
          <span className="flex items-center gap-2">
            <Play className="w-4 h-4 fill-current" />
            Run Simulation
          </span>
        )}
      </Button>
    </form>
  );
}
