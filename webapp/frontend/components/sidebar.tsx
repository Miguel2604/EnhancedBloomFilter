"use client";

import { LayoutDashboard, Activity } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="w-20 bg-white h-screen flex flex-col items-center py-8 border-r border-gray-100 fixed left-0 top-0 z-50">

      
      <nav className="flex flex-col gap-4 w-full px-4">
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
      </nav>


    </div>
  );
}

function NavItem({ href, icon, active, label }: { href: string; icon: React.ReactNode; active?: boolean; label: string }) {
  return (
    <Link href={href} className="group relative flex justify-center">
      <button
        className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300 ${
          active 
            ? "bg-black text-white shadow-lg shadow-gray-300 scale-105" 
            : "text-gray-400 hover:bg-gray-100 hover:text-gray-900"
        }`}
      >
        <div className="w-5 h-5 [&>svg]:w-full [&>svg]:h-full">{icon}</div>
      </button>
      
      {/* Tooltip */}
      <div className="absolute left-14 top-1/2 -translate-y-1/2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
        {label}
      </div>
    </Link>
  );
}
