import { cn } from "@/lib/utils";

interface DashboardCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  action?: React.ReactNode;
}

export function DashboardCard({ title, action, children, className, ...props }: DashboardCardProps) {
  return (
    <div 
      className={cn(
        "bg-white rounded-none p-6 border border-black flex flex-col h-full transition-all hover:bg-gray-50", 
        className
      )} 
      {...props}
    >
      {(title || action) && (
        <div className="flex items-center justify-between mb-6 pb-4 border-b border-black">
          {title && <h3 className="text-xl font-bold uppercase tracking-tight text-black">{title}</h3>}
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="flex-1">{children}</div>
    </div>
  );
}
