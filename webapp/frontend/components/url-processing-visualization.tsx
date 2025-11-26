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
      <Card className="lg:col-span-3 flex flex-col h-[450px] rounded-none border-black shadow-none">
        <CardHeader className="pb-2 space-y-4 shrink-0 border-b border-black">
          <div className="space-y-1.5">
            <CardTitle className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest">
              <ListVideo className="w-5 h-5 text-black" />
              Input Queue
            </CardTitle>
            <CardDescription className="font-mono text-xs">Dataset Stream ({queue.length} items)</CardDescription>
          </div>
          
          <div className="flex flex-col gap-2">
            <Button 
              className={`w-full rounded-none border border-black ${isStreaming ? 'bg-red-600 hover:bg-red-700 text-white' : 'bg-black text-white'}`}
              onClick={toggleStream}
              disabled={queue.length === 0}
            >
              {isStreaming ? (
                <><Pause className="w-4 h-4 mr-2" /> STOP STREAM</>
              ) : (
                <><Play className="w-4 h-4 mr-2" /> START STREAM</>
              )}
            </Button>
            
            <div className="flex items-center justify-between bg-white p-2 border border-black">
              <span className="text-xs font-bold uppercase text-black">Speed</span>
              <div className="flex gap-1">
                {([1, 2, 5] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSpeed(s)}
                    className={`px-2 py-1 text-xs font-mono font-bold transition-colors border border-transparent ${
                      speed === s 
                        ? 'bg-black text-white border-black' 
                        : 'text-gray-500 hover:bg-gray-200 hover:text-black'
                    }`}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="flex-1 min-h-0 p-0">
          <ScrollArea className="h-full">
            <div className="space-y-0 p-0">
              {queue.map((item, i) => (
                <div key={i} className="flex items-center gap-3 p-3 border-b border-gray-200 text-sm animate-in fade-in slide-in-from-right-4 duration-300">
                  <div className={`w-3 h-3 rounded-none border border-black shrink-0 ${
                    item.type === 'safe' ? 'bg-white' : 
                    item.type === 'malicious' ? 'bg-red-600' : 'bg-gray-400'
                  }`} />
                  <span className="truncate font-bold font-mono text-black">{item.url}</span>
                </div>
              ))}
              {queue.length === 0 && (
                <div className="text-center text-gray-400 py-8">
                  <p className="uppercase font-bold text-xs">Queue Empty</p>
                  <Button variant="link" onClick={fetchDataset} className="mt-2 text-black font-bold uppercase underline">
                    Reload Dataset
                  </Button>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Center: Processing Pipeline */}
      <Card className="lg:col-span-6 flex flex-col relative overflow-hidden min-h-[500px] rounded-none border-black shadow-none">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:20px_20px] opacity-20 pointer-events-none" />
        <CardHeader className="border-b border-black bg-white shrink-0">
          <CardTitle className="flex items-center justify-between">
             <span className="text-sm font-bold uppercase tracking-widest">Processing Pipeline</span>
             {currentItem && (
               <Badge variant="outline" className="rounded-none border-black bg-black text-white font-mono uppercase">
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
                 <Activity className="w-16 h-16 opacity-20 mb-4 text-black" />
                 <p className="uppercase font-bold tracking-widest">Ready to process</p>
               </div>
            ) : (
               currentSteps.map((step, index) => (
                <div 
                  key={index} 
                  className={`flex items-start gap-4 transition-all duration-300 ${
                    step.status === "active" ? "opacity-100 translate-x-2" : "opacity-60"
                  }`}
                >
                  <div className={`mt-1 w-8 h-8 rounded-none flex items-center justify-center border-2 shrink-0 ${
                    step.status === "active" ? "bg-white border-black text-black" :
                    step.status === "completed" ? "bg-black border-black text-white" :
                    "bg-gray-100 border-gray-300 text-gray-400"
                  }`}>
                    {getStepIcon(step.stage)}
                  </div>
                  
                  <div className={`flex-1 p-4 rounded-none border ${
                    step.status === "active" ? "bg-white border-black ring-0" :
                    "bg-white border-gray-200"
                  }`}>
                    <div className="flex justify-between items-start mb-1">
                      <h4 className="font-bold text-black uppercase tracking-wide text-sm">{step.stage}</h4>
                      {step.status === "active" && <span className="text-xs text-black font-bold uppercase bg-gray-200 px-1">Processing</span>}
                    </div>
                    <p className="text-sm text-gray-600 mb-2 font-mono">{step.description}</p>
                    
                    {/* Compact Details */}
                    {step.details && (
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(step.details).map(([k, v]) => (
                          (typeof v !== 'object' && v !== null) && (
                            <Badge key={k} variant="secondary" className="rounded-none text-xs font-mono font-normal border border-gray-300 bg-white text-black">
                              {k}: {String(v)}
                            </Badge>
                          )
                        ))}
                        {step.details.decision && (
                             <Badge className={`rounded-none border border-black ${step.details.decision === 'POSITIVE' ? 'bg-black text-white' : 'bg-red-600 text-white'}`}>
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
      <Card className="lg:col-span-3 flex flex-col h-[450px] rounded-none border-black shadow-none">
        <CardHeader className="pb-2 shrink-0 border-b border-black">
          <CardTitle className="text-sm font-bold uppercase tracking-widest">Processed Log</CardTitle>
          <CardDescription className="font-mono text-xs">Real-time decisions</CardDescription>
        </CardHeader>
        <CardContent className="flex-1 min-h-0 p-0 overflow-hidden">
           <ScrollArea className="h-full">
             <div className="space-y-0 p-0">
               {logs.map((log, i) => (
                 <div key={log.timestamp} className="p-3 bg-white border-b border-gray-200 animate-in slide-in-from-top-2 fade-in duration-300 hover:bg-gray-50">
                   <div className="flex items-center justify-between mb-1">
                     <Badge variant={log.result ? "default" : "secondary"} className={`rounded-none border ${!log.result ? "bg-white text-black border-black" : "bg-black text-white border-black"}`}>
                       {log.result ? "POSITIVE" : "NEGATIVE"}
                     </Badge>
                     <span className="text-[10px] text-black font-mono font-bold">
                       {new Date(log.timestamp).toLocaleTimeString().split(' ')[0]}
                     </span>
                   </div>
                   <div className="font-bold font-mono text-sm truncate mb-1 text-black" title={log.url}>{log.url}</div>
                   <div className="flex items-center gap-1 text-xs text-gray-500 font-bold uppercase">
                     <ArrowRight className="w-3 h-3" />
                     {log.path}
                   </div>
                 </div>
               ))}
               {logs.length === 0 && (
                 <div className="text-center text-gray-400 py-12 text-sm uppercase font-bold">
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
