import { cn } from "@/lib/utils";

interface DashboardCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  action?: React.ReactNode;
}

export function DashboardCard({ title, action, children, className, ...props }: DashboardCardProps) {
  return (
    <div 
      className={cn(
        "bg-white rounded-[2rem] p-6 shadow-sm border border-gray-100 flex flex-col h-full transition-all hover:shadow-md", 
        className
      )} 
      {...props}
    >
      {(title || action) && (
        <div className="flex items-center justify-between mb-6">
          {title && <h3 className="text-lg font-semibold text-gray-900">{title}</h3>}
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="flex-1">{children}</div>
    </div>
  );
}
