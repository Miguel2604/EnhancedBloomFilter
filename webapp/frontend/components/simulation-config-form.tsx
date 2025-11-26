import { SimulationConfig, FILTER_OPTIONS } from "@/types/simulation";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
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
      <div className="space-y-6">
        <div className="space-y-3">
          <Label className="text-black font-bold uppercase tracking-wider text-xs">
            Dataset Size: {config.datasetSize.toLocaleString()} items
          </Label>
          <Slider
            value={[config.datasetSize]}
            onValueChange={(val) => updateConfig({ datasetSize: val[0] })}
            min={10000}
            max={200000}
            step={10000}
            className="w-full"
          />
        </div>
        <div className="space-y-3">
          <Label className="text-black font-bold uppercase tracking-wider text-xs">
            Query Count: {config.queryCount.toLocaleString()} queries
          </Label>
          <Slider
            value={[config.queryCount]}
            onValueChange={(val) => updateConfig({ queryCount: val[0] })}
            min={10000}
            max={100000}
            step={5000}
            className="w-full"
          />
        </div>
      </div>

      <div className="space-y-3">
        <Label className="text-black font-bold uppercase tracking-wider text-xs">Active Filters</Label>
        <div className="space-y-0 border border-black">
          {FILTER_OPTIONS.map((filter, index) => (
            <div
              key={filter.id}
              className={`flex items-center space-x-3 p-3 cursor-pointer border-b last:border-b-0 border-black transition-colors ${
                selectedFilters.includes(filter.id)
                  ? "bg-black text-white"
                  : "bg-white hover:bg-gray-100 text-black"
              }`}
              onClick={() => toggleFilter(filter.id)}
            >
              <div
                className={`w-4 h-4 rounded-none flex items-center justify-center border ${
                  selectedFilters.includes(filter.id)
                    ? "border-white bg-black"
                    : "border-black bg-white"
                }`}
              >
                {selectedFilters.includes(filter.id) && (
                  <div className="w-2 h-2 bg-white rounded-none" />
                )}
              </div>
              <span className="text-sm font-bold uppercase tracking-wide">
                {filter.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      <Button 
        type="submit" 
        disabled={isLoading || selectedFilters.length === 0}
        className="w-full rounded-none h-12 text-sm font-bold uppercase tracking-widest bg-black hover:bg-gray-900 text-white shadow-none border border-black"
      >
        {isLoading ? (
          <span className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-white border-t-transparent animate-spin" />
            Processing...
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
