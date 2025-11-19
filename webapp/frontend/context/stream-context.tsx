"use client";

import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from "react";

// Types
interface Step {
  stage: string;
  description: string;
  details: any;
  status: "pending" | "active" | "completed";
}

interface ProcessResponse {
  url: string;
  filter_type: string;
  steps: Step[];
  final_result: boolean;
}

interface DatasetItem {
  url: string;
  type: "safe" | "malicious" | "unknown";
}

interface LogItem {
  url: string;
  result: boolean;
  path: "Fast Path" | "Cache" | "Backup";
  timestamp: number;
}

interface StreamContextType {
  queue: DatasetItem[];
  logs: LogItem[];
  isStreaming: boolean;
  speed: 1 | 2 | 5;
  currentItem: DatasetItem | null;
  currentSteps: Step[];
  
  setSpeed: (speed: 1 | 2 | 5) => void;
  toggleStream: () => void;
  fetchDataset: () => Promise<void>;
}

const StreamContext = createContext<StreamContextType | undefined>(undefined);

export function StreamProvider({ children }: { children: React.ReactNode }) {
  // State
  const [queue, setQueue] = useState<DatasetItem[]>([]);
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [speed, setSpeed] = useState<1 | 2 | 5>(1);
  const [isProcessingItem, setIsProcessingItem] = useState(false);
  
  const [currentItem, setCurrentItem] = useState<DatasetItem | null>(null);
  const [currentSteps, setCurrentSteps] = useState<Step[]>([]);
  
  const processingRef = useRef(false);

  // Fetch dataset
  const fetchDataset = useCallback(async () => {
    try {
      const res = await fetch("http://127.0.0.1:5000/api/dataset");
      const data = await res.json();
      setQueue(data.items || []);
    } catch (err) {
      console.error("Failed to fetch dataset", err);
    }
  }, []);

  // Initial fetch if empty (optional, maybe triggered by user)
  // We won't auto-fetch here to avoid double fetch issues, letting component do it or user.

  const processNextItem = useCallback(async () => {
    if (processingRef.current || queue.length === 0) return;
    
    processingRef.current = true;
    setIsProcessingItem(true);
    
    const item = queue[0];
    setQueue(prev => prev.slice(1));
    setCurrentItem(item);
    setCurrentSteps([]);

    try {
      const res = await fetch("http://127.0.0.1:5000/api/process-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: item.url, filter_type: "Combined Enhanced LBF" }),
      });
      
      const data: ProcessResponse = await res.json();
      
      let path: LogItem["path"] = "Backup";
      if (data.steps.some(s => s.stage === "Fast Path")) path = "Fast Path";
      else if (data.steps.some(s => s.stage === "Cache Check" && s.details?.conclusion?.includes("Definitely"))) path = "Cache";

      // Calculate delay based on speed
      const stepDelay = 600 / speed;
      
      for (let i = 0; i < data.steps.length; i++) {
        if (!processingRef.current) break;
        
        setCurrentSteps(data.steps.map((s, idx) => ({
          ...s,
          status: idx < i ? "completed" : idx === i ? "active" : "pending"
        })));
        
        await new Promise(r => setTimeout(r, stepDelay));
      }

      if (!processingRef.current) return;

      setCurrentSteps(data.steps.map(s => ({ ...s, status: "completed" })));
      
      setLogs(prev => [{
        url: item.url,
        result: data.final_result,
        path: path,
        timestamp: Date.now()
      }, ...prev].slice(0, 50)); // Keep last 50

      await new Promise(r => setTimeout(r, 200 / speed));

    } catch (err) {
      console.error("Error processing item", err);
    } finally {
      processingRef.current = false;
      setIsProcessingItem(false);
      setCurrentItem(null);
    }
  }, [queue, speed]);

  // Processing Loop
  useEffect(() => {
    if (isStreaming && !isProcessingItem && queue.length > 0) {
      processNextItem();
    } else if (isStreaming && queue.length === 0 && !isProcessingItem) {
      setIsStreaming(false);
    }
  }, [isStreaming, isProcessingItem, queue, processNextItem]);

  const toggleStream = useCallback(() => {
    if (isStreaming) {
      setIsStreaming(false);
      processingRef.current = false;
    } else {
      setIsStreaming(true);
    }
  }, [isStreaming]);

  return (
    <StreamContext.Provider value={{
      queue,
      logs,
      isStreaming,
      speed,
      currentItem,
      currentSteps,
      setSpeed,
      toggleStream,
      fetchDataset
    }}>
      {children}
    </StreamContext.Provider>
  );
}

export function useStreamContext() {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
}
