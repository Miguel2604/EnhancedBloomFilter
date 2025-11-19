"use client";

import { FilterResult, FILTER_OPTIONS } from "@/types/simulation";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";

interface MetricsComparisonProps {
  results: FilterResult[];
  metric: "throughput" | "memoryBytes" | "creationTimeMs";
  unit?: string;
  formatter?: (value: number) => string;
}

export function MetricsComparison({ results, metric, unit, formatter }: MetricsComparisonProps) {
  return (
    <div className="h-[200px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={results} layout="vertical" margin={{ top: 0, right: 80, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#f0f0f0" />
          <XAxis type="number" hide />
          <YAxis 
            dataKey="label" 
            type="category" 
            width={100} 
            axisLine={false} 
            tickLine={false}
            tick={{ fill: '#6b7280', fontSize: 12, fontWeight: 500 }}
          />
          <Tooltip 
            cursor={{ fill: '#f9fafb' }}
            contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
            formatter={(value: number) => [formatter ? formatter(value) : value, unit]}
          />
          <Bar dataKey={metric} radius={[0, 4, 4, 0]} barSize={20}>
            <LabelList 
              dataKey={metric} 
              position="right" 
              formatter={formatter}
              style={{ fontSize: 12, fill: '#374151', fontWeight: 500 }}
            />
            {results.map((entry, index) => {
               const color = FILTER_OPTIONS.find(o => o.id === entry.id)?.color || "#000";
               return <Cell key={`cell-${index}`} fill={color} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
