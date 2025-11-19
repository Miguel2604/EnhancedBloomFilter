"use client";

import { useStreamContext } from "@/context/stream-context";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, Pause, RefreshCw, CheckCircle, ArrowRight, Server, Cpu, Database, ShieldCheck, Activity, ListVideo, FastForward } from 'lucide-react';
import { ScrollArea } from "@/components/ui/scroll-area";

export function UrlProcessingVisualization() {
  const {
    queue,
    logs,
    isStreaming,
    speed,
    currentItem,
    currentSteps,
    setSpeed,
    toggleStream,
    fetchDataset
  } = useStreamContext();

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      
      {/* Left Column: Queue */}
      <Card className="lg:col-span-3 flex flex-col h-[450px]">
        <CardHeader className="pb-2 space-y-4 shrink-0">
          <div className="space-y-1.5">
            <CardTitle className="flex items-center gap-2 text-lg">
              <ListVideo className="w-5 h-5 text-purple-500" />
              Input Queue
            </CardTitle>
            <CardDescription>Dataset Stream ({queue.length} items)</CardDescription>
          </div>
          
          <div className="flex flex-col gap-2">
            <Button 
              className={`w-full ${isStreaming ? 'bg-red-500 hover:bg-red-600' : 'bg-black'}`}
              onClick={toggleStream}
              disabled={queue.length === 0}
            >
              {isStreaming ? (
                <><Pause className="w-4 h-4 mr-2" /> Stop Stream</>
              ) : (
                <><Play className="w-4 h-4 mr-2" /> Start Stream</>
              )}
            </Button>
            
            <div className="flex items-center justify-between bg-gray-50 p-2 rounded-lg border border-gray-100">
              <span className="text-xs font-medium text-gray-500">Speed</span>
              <div className="flex gap-1">
                {([1, 2, 5] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSpeed(s)}
                    className={`px-2 py-1 text-xs rounded transition-colors ${
                      speed === s 
                        ? 'bg-white text-black font-bold shadow-sm border border-gray-200' 
                        : 'text-gray-500 hover:bg-gray-200 hover:text-gray-700'
                    }`}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="flex-1 min-h-0 p-0 border-t border-gray-100">
          <ScrollArea className="h-full">
            <div className="space-y-2 p-4">
              {queue.map((item, i) => (
                <div key={i} className="flex items-center gap-3 p-3 bg-gray-50 rounded-xl border border-gray-100 text-sm animate-in fade-in slide-in-from-right-4 duration-300">
                  <div className={`w-2 h-2 rounded-full shrink-0 ${
                    item.type === 'safe' ? 'bg-green-400' : 
                    item.type === 'malicious' ? 'bg-red-400' : 'bg-gray-400'
                  }`} />
                  <span className="truncate font-medium text-gray-700">{item.url}</span>
                </div>
              ))}
              {queue.length === 0 && (
                <div className="text-center text-gray-400 py-8">
                  <p>Queue Empty</p>
                  <Button variant="link" onClick={fetchDataset} className="mt-2 text-purple-600">
                    Reload Dataset
                  </Button>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Center: Processing Pipeline */}
      <Card className="lg:col-span-6 flex flex-col relative overflow-hidden min-h-[500px]">
        <div className="absolute inset-0 bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:16px_16px] opacity-20 pointer-events-none" />
        <CardHeader className="border-b bg-white/80 backdrop-blur shrink-0">
          <CardTitle className="flex items-center justify-between">
             <span>Processing Pipeline</span>
             {currentItem && (
               <Badge variant="outline" className="animate-pulse border-purple-200 bg-purple-50 text-purple-700">
                 Processing: {currentItem.url}
               </Badge>
             )}
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col justify-center p-8 relative min-h-0 overflow-y-auto">
          
          {/* Pipeline Visualization */}
          <div className="space-y-6 relative">
            {!currentItem ? (
               <div className="flex flex-col items-center justify-center text-gray-400 py-12">
                 <Activity className="w-16 h-16 opacity-20 mb-4" />
                 <p>Ready to process</p>
               </div>
            ) : (
               currentSteps.map((step, index) => (
                <div 
                  key={index} 
                  className={`flex items-start gap-4 transition-all duration-300 ${
                    step.status === "active" ? "opacity-100 scale-105" : "opacity-60"
                  }`}
                >
                  <div className={`mt-1 w-8 h-8 rounded-full flex items-center justify-center shadow-sm border-2 shrink-0 ${
                    step.status === "active" ? "bg-white border-blue-500 text-blue-500 animate-pulse" :
                    step.status === "completed" ? "bg-green-500 border-green-500 text-white" :
                    "bg-gray-100 border-gray-200 text-gray-400"
                  }`}>
                    {getStepIcon(step.stage)}
                  </div>
                  
                  <div className={`flex-1 p-4 rounded-2xl border ${
                    step.status === "active" ? "bg-white border-blue-200 shadow-lg ring-4 ring-blue-50" :
                    "bg-white border-gray-100"
                  }`}>
                    <div className="flex justify-between items-start mb-1">
                      <h4 className="font-bold text-gray-900">{step.stage}</h4>
                      {step.status === "active" && <span className="text-xs text-blue-500 font-medium">Processing...</span>}
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{step.description}</p>
                    
                    {/* Compact Details */}
                    {step.details && (
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(step.details).map(([k, v]) => (
                          (typeof v !== 'object' && v !== null) && (
                            <Badge key={k} variant="secondary" className="text-xs font-mono font-normal">
                              {k}: {String(v)}
                            </Badge>
                          )
                        ))}
                        {step.details.decision && (
                             <Badge className={step.details.decision === 'POSITIVE' ? 'bg-green-500' : 'bg-red-500'}>
                                {step.details.decision}
                             </Badge>
                        )}
                      </div>
                    )}
                  </div>
                </div>
               ))
            )}
          </div>

        </CardContent>
      </Card>

      {/* Right Column: Logs */}
      <Card className="lg:col-span-3 flex flex-col h-[450px]">
        <CardHeader className="pb-2 shrink-0">
          <CardTitle className="text-lg">Processed Log</CardTitle>
          <CardDescription>Real-time decisions</CardDescription>
        </CardHeader>
        <CardContent className="flex-1 min-h-0 p-0 overflow-hidden">
           <ScrollArea className="h-full">
             <div className="space-y-3 p-4">
               {logs.map((log, i) => (
                 <div key={log.timestamp} className="p-3 bg-white rounded-xl border border-gray-100 shadow-sm animate-in slide-in-from-top-2 fade-in duration-300">
                   <div className="flex items-center justify-between mb-1">
                     <Badge variant={log.result ? "default" : "secondary"} className={!log.result ? "bg-gray-100 text-gray-500" : "bg-black"}>
                       {log.result ? "Positive" : "Negative"}
                     </Badge>
                     <span className="text-[10px] text-gray-400 font-mono">
                       {new Date(log.timestamp).toLocaleTimeString().split(' ')[0]}
                     </span>
                   </div>
                   <div className="font-medium text-sm truncate mb-1" title={log.url}>{log.url}</div>
                   <div className="flex items-center gap-1 text-xs text-gray-500">
                     <ArrowRight className="w-3 h-3" />
                     {log.path}
                   </div>
                 </div>
               ))}
               {logs.length === 0 && (
                 <div className="text-center text-gray-400 py-12 text-sm">
                   Results will appear here
                 </div>
               )}
             </div>
           </ScrollArea>
        </CardContent>
      </Card>
    </div>
  );
}

function getStepIcon(stage: string) {
  const lower = stage.toLowerCase();
  if (lower.includes("feature")) return <Cpu className="w-4 h-4" />;
  if (lower.includes("model")) return <Server className="w-4 h-4" />;
  if (lower.includes("cache")) return <Database className="w-4 h-4" />;
  if (lower.includes("backup")) return <ShieldCheck className="w-4 h-4" />;
  if (lower.includes("result") || lower.includes("fast")) return <CheckCircle className="w-4 h-4" />;
  return <Activity className="w-4 h-4" />;
}
