import { Link, useLocation } from "react-router-dom";
import { FileText, Search, BarChart2, BookOpen, Settings } from "lucide-react";

export function Sidebar() {
  const location = useLocation();

  const links = [
    { to: "/", icon: <BookOpen className="w-5 h-5" />, label: "Overview" },
    { to: "/documents", icon: <FileText className="w-5 h-5" />, label: "Documents" },
    { to: "/query", icon: <Search className="w-5 h-5" />, label: "Query" },
    { to: "/experiments", icon: <BarChart2 className="w-5 h-5" />, label: "Experiments" },
    { to: "/settings", icon: <Settings className="w-5 h-5" />, label: "Settings" },
  ];

  return (
    <aside className="w-64 bg-gray-900 text-white min-h-screen p-4 flex flex-col">
      <div className="mb-8 px-2">
        <h1 className="text-xl font-bold tracking-wider">ResearchGuard AI</h1>
        <p className="text-xs text-gray-400 mt-1">Claim Verification System</p>
      </div>
      
      <nav className="flex-1 space-y-1">
        {links.map((link) => {
          const active = location.pathname === link.to || (link.to !== "/" && location.pathname.startsWith(link.to));
          return (
            <Link
              key={link.to}
              to={link.to}
              className={`flex items-center space-x-3 px-3 py-2 rounded-md transition-colors ${
                active ? "bg-blue-600 text-white" : "text-gray-300 hover:bg-gray-800 hover:text-white"
              }`}
            >
              {link.icon}
              <span className="font-medium">{link.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
