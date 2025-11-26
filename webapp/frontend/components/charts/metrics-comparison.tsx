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
  // Ensure we have valid numbers for the chart domain
  const maxVal = results.length > 0 ? Math.max(...results.map(r => r[metric] || 0)) : 0;
  
  // Calculate dynamic width for Y-axis based on label length (approx 7px per char)
  const maxLabelLen = results.length > 0 ? Math.max(...results.map(r => (r.label || "").length)) : 0;
  const yAxisWidth = Math.max(100, maxLabelLen * 7 + 20);

  return (
    <div className="h-[200px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={results} layout="vertical" margin={{ top: 0, right: 80, left: 0, bottom: 0 }}>
          <CartesianGrid horizontal={true} vertical={false} stroke="#e5e5e5" />
          <XAxis type="number" hide domain={[0, maxVal * 1.2]} />
          <YAxis 
            dataKey="label" 
            type="category" 
            width={yAxisWidth} 
            axisLine={false} 
            tickLine={false}
            tick={{ fill: '#000', fontSize: 10, fontWeight: 'bold', fontFamily: 'sans-serif' }}
          />
          <Tooltip 
            cursor={{ fill: '#f4f4f4' }}
            contentStyle={{ borderRadius: '0px', border: '1px solid black', boxShadow: 'none', fontFamily: 'sans-serif' }}
            formatter={(value: number) => [formatter ? formatter(value) : value, unit]}
            itemStyle={{ fontWeight: 'bold', color: 'black' }}
          />
          <Bar dataKey={metric} radius={[0, 0, 0, 0]} barSize={20}>
            <LabelList 
              dataKey={metric} 
              position="right" 
              formatter={formatter ? (value) => formatter(Number(value)) : undefined}
              style={{ fontSize: 10, fill: '#000', fontWeight: 'bold', fontFamily: 'sans-serif' }}
            />
            {results.map((entry, index) => {
               const colors = ["#000000", "#FF3333", "#666666", "#999999", "#CCCCCC"];
               const color = colors[index % colors.length];
               return <Cell key={`cell-${index}`} fill={color} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
