import { LayoutDashboard, Settings, BarChart3, Database, Home, PieChart, Layers } from 'lucide-react';

export function Sidebar() {
  return (
    <div className="w-20 bg-white h-screen flex flex-col items-center py-8 border-r border-gray-100 fixed left-0 top-0 z-10">
      <div className="w-10 h-10 bg-black rounded-xl flex items-center justify-center text-white font-bold text-xl mb-12">
        S
      </div>
      
      <nav className="flex flex-col gap-8">
        <NavItem icon={<LayoutDashboard />} active />
        <NavItem icon={<BarChart3 />} />
        <NavItem icon={<Layers />} />
        <NavItem icon={<Database />} />
        <NavItem icon={<PieChart />} />
      </nav>

      <div className="mt-auto flex flex-col gap-8">
        <NavItem icon={<Settings />} />
      </div>
    </div>
  );
}

function NavItem({ icon, active }: { icon: React.ReactNode; active?: boolean }) {
  return (
    <button
      className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all ${
        active 
          ? "bg-black text-white shadow-lg shadow-gray-200" 
          : "text-gray-400 hover:bg-gray-50 hover:text-gray-600"
      }`}
    >
      {/* Clone element to enforce size if needed, but lucide icons usually scale well */}
      <div className="w-5 h-5 [&>svg]:w-full [&>svg]:h-full">{icon}</div>
    </button>
  );
}
