export function Header() {
  return (
    <header className="bg-white border-b h-16 flex items-center px-6 justify-between">
      <h2 className="text-lg font-semibold text-gray-800">Dashboard</h2>
      <div className="flex items-center space-x-4">
        <div className="text-sm text-gray-500">Local Environment</div>
        <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-bold">
          RG
        </div>
      </div>
    </header>
  );
}
