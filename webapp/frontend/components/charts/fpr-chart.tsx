"use client";

import { FilterResult } from "@/types/simulation";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

// Helper to calculate standard deviation
function calculateStdDev(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
  const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(avgSquaredDiff);
}

interface FprChartProps {
  results: FilterResult[];
}

export function FprChart({ results }: FprChartProps) {
  // Transform data for Recharts: array of objects with step and values for each filter
  const dataLength = results[0]?.fprHistory.length || 0;
  const data = Array.from({ length: dataLength }, (_, i) => {
    const point: Record<string, number> = { step: i };
    results.forEach((r) => {
      point[r.id] = r.fprHistory[i];
    });
    return point;
  });

  // Calculate stats for each filter
  const filterStats = results.map((r) => {
    const mean = r.fprHistory.reduce((a, b) => a + b, 0) / r.fprHistory.length;
    const stdDev = calculateStdDev(r.fprHistory);
    return { id: r.id, label: r.label, mean, stdDev };
  });

  // Define colors for each filter
  const colors = ["#000000", "#FF3333", "#666666", "#999999", "#CCCCCC"];

  return (
    <div className="space-y-4">
      {/* FPR Statistics Table */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {filterStats.map((stat, index) => (
          <div key={stat.id} className="p-3 border border-gray-200 bg-gray-50">
            <div className="flex items-center gap-2 mb-2">
              <div 
                className="w-3 h-3" 
                style={{ backgroundColor: colors[index % colors.length] }}
              />
              <span className="text-xs font-bold uppercase tracking-wider truncate">
                {stat.label}
              </span>
            </div>
            <div className="font-mono text-xs space-y-1">
              <div>
                <span className="text-gray-500">Mean:</span>{" "}
                <span className="font-bold">{(stat.mean * 100).toFixed(3)}%</span>
              </div>
              <div>
                <span className="text-gray-500">Std Dev:</span>{" "}
                <span className="font-bold" style={{ color: colors[index % colors.length] }}>
                  {(stat.stdDev * 100).toFixed(4)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid stroke="#e5e5e5" vertical={false} />
          <XAxis 
            dataKey="step" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#000', fontSize: 10, fontFamily: 'sans-serif', fontWeight: 'bold' }}
            dy={10}
          />
          <YAxis 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#000', fontSize: 10, fontFamily: 'sans-serif', fontWeight: 'bold' }}
            tickFormatter={(value) => value.toFixed(3)}
          />
          <Tooltip 
            contentStyle={{ borderRadius: '0px', border: '1px solid black', boxShadow: 'none', backgroundColor: 'white' }}
            itemStyle={{ fontFamily: 'sans-serif', fontSize: '12px', fontWeight: 'bold' }}
          />
          <Legend wrapperStyle={{ fontFamily: 'sans-serif', fontSize: '12px', textTransform: 'uppercase', fontWeight: 'bold' }} iconType="square" />
          {results.map((r, index) => {
            const color = colors[index % colors.length];
            return (
              <Line
                key={r.id}
                type="linear"
                dataKey={r.id}
                name={r.label}
                stroke={color}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 0, fill: color }}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}
