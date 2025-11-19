"use client";

import { FilterResult, FILTER_OPTIONS } from "@/types/simulation";
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

interface FprChartProps {
  results: FilterResult[];
}

export function FprChart({ results }: FprChartProps) {
  // Transform data for Recharts: array of objects with step and values for each filter
  const dataLength = results[0]?.fprHistory.length || 0;
  const data = Array.from({ length: dataLength }, (_, i) => {
    const point: any = { step: i };
    results.forEach((r) => {
      point[r.id] = r.fprHistory[i];
    });
    return point;
  });

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
          <XAxis 
            dataKey="step" 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            dy={10}
          />
          <YAxis 
            axisLine={false} 
            tickLine={false} 
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickFormatter={(value) => value.toFixed(3)}
          />
          <Tooltip 
            contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
          />
          <Legend iconType="circle" />
          {results.map((r) => {
            const color = FILTER_OPTIONS.find(o => o.id === r.id)?.color || "#000";
            return (
              <Line
                key={r.id}
                type="monotone"
                dataKey={r.id}
                name={r.label}
                stroke={color}
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 6, strokeWidth: 0 }}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
