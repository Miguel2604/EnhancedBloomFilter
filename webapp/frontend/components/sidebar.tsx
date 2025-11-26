"use client";

import { LayoutDashboard, Activity, GitCompare } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="w-20 bg-white h-screen flex flex-col items-center py-8 border-r border-black fixed left-0 top-0 z-50">

      
      <nav className="flex flex-col gap-0 w-full">
        <NavItem 
          href="/" 
          icon={<LayoutDashboard />} 
          label="Dashboard"
          active={pathname === "/"} 
        />
        <NavItem 
          href="/realtime" 
          icon={<Activity />} 
          label="Realtime"
          active={pathname === "/realtime"} 
        />
        <NavItem 
          href="/comparison" 
          icon={<GitCompare />} 
          label="Comparison"
          active={pathname === "/comparison"} 
        />
      </nav>


    </div>
  );
}

function NavItem({ href, icon, active, label }: { href: string; icon: React.ReactNode; active?: boolean; label: string }) {
  return (
    <Link href={href} className="group relative flex justify-center w-full">
      <button
        className={`w-full h-16 flex items-center justify-center transition-none border-b border-transparent ${
          active 
            ? "bg-black text-white" 
            : "text-gray-400 hover:bg-gray-100 hover:text-black hover:border-black"
        }`}
      >
        <div className="w-6 h-6 [&>svg]:w-full [&>svg]:h-full">{icon}</div>
      </button>
      
      {/* Tooltip */}
      <div className="absolute left-full top-0 h-16 flex items-center px-4 bg-black text-white text-xs font-bold uppercase tracking-wider opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50 border-l border-white">
        {label}
      </div>
    </Link>
  );
}
