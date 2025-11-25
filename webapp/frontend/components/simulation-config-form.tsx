import { SimulationConfig, FILTER_OPTIONS } from "@/types/simulation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Play } from 'lucide-react';

interface SimulationConfigFormProps {
  onSubmit: (config: SimulationConfig) => void;
  isLoading: boolean;
  config: SimulationConfig;
  onConfigChange: (config: SimulationConfig) => void;
}

export function SimulationConfigForm({ onSubmit, isLoading, config, onConfigChange }: SimulationConfigFormProps) {
  const selectedFilters = config.selectedFilters;
  const updateConfig = (partial: Partial<SimulationConfig>) => {
    onConfigChange({ ...config, ...partial });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(config);
  };

  const toggleFilter = (id: string) => {
    const next = selectedFilters.includes(id)
      ? selectedFilters.filter((f) => f !== id)
      : [...selectedFilters, id];
    updateConfig({ selectedFilters: next });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="dataset-size" className="text-gray-500 font-medium">Dataset Size</Label>
          <Input
            id="dataset-size"
            type="number"
            value={config.datasetSize}
            onChange={(e) => updateConfig({ datasetSize: Number(e.target.value) })}
            className="rounded-xl border-gray-200 bg-gray-50 focus:bg-white transition-colors"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="query-count" className="text-gray-500 font-medium">Query Count</Label>
          <Input
            id="query-count"
            type="number"
            value={config.queryCount}
            onChange={(e) => updateConfig({ queryCount: Number(e.target.value) })}
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
