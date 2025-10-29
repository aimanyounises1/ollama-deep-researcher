import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

function Header() {
  const navigate = useNavigate();
  const location = useLocation();
  
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  
  const handleMobileMenuToggle = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };
  
  const handleUserMenuToggle = () => {
    setUserMenuOpen(!userMenuOpen);
  };
  
  const handleModelMenuToggle = () => {
    setModelMenuOpen(!modelMenuOpen);
  };
  
  const handleNavigation = (path) => {
    navigate(path);
    setMobileMenuOpen(false);
  };
  
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  // Navigation items
  const navItems = [
    { name: 'Chat', path: '/' },
    { name: 'Research', path: '/research' },
    { name: 'Quiz', path: '/quiz' },
    { name: 'Home', path: '/home' },
  ];
  
  return (
    <header className="sticky top-0 z-50 bg-gray-900/90 backdrop-blur-md border-b border-gray-800 shadow-md">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={handleMobileMenuToggle}
              className="text-gray-300 hover:text-white p-2"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
          
          {/* Logo and title */}
          <div className="flex items-center">
            <div className="h-9 w-9 mr-2 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold text-lg">
              V
            </div>
            <div className="text-xl font-bold bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
              VFIT Deep Research
            </div>
          </div>
          
          {/* Desktop navigation */}
          <div className="hidden md:flex space-x-1 flex-1 ml-10">
            {navItems.map((item) => (
              <button
                key={item.name}
                onClick={() => handleNavigation(item.path)}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive(item.path)
                    ? 'bg-blue-900/30 text-blue-400'
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                }`}
              >
                {item.name}
              </button>
            ))}
          </div>
          
          {/* Right section - model selector and user menu */}
          <div className="flex items-center space-x-4">
            {/* Model selector */}
            <div className="relative">
              <button
                onClick={handleModelMenuToggle}
                className="flex items-center px-3 py-1.5 bg-blue-900/20 rounded-lg text-sm text-gray-200 hover:bg-blue-900/30 transition-colors"
              >
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2 shadow-glow-green"></div>
                  <span className="font-medium">Mixtral 8x7B</span>
                  <svg className="w-4 h-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>
              
              {/* Model dropdown menu */}
              {modelMenuOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-gray-800 border border-gray-700 rounded-lg shadow-lg py-1 z-10">
                  <div className="px-4 py-2 bg-blue-900/20 text-sm font-medium text-white flex items-center">
                    <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                    Mixtral 8x7B
                    <span className="text-xs text-gray-400 ml-2">Default</span>
                  </div>
                  <div className="px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 flex items-center">
                    <div className="w-2 h-2 bg-gray-500 rounded-full mr-2"></div>
                    Llama 2 13B
                  </div>
                  <div className="px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 flex items-center">
                    <div className="w-2 h-2 bg-gray-500 rounded-full mr-2"></div>
                    Llama 2 70B
                  </div>
                  <div className="border-t border-gray-700 mt-1"></div>
                  <div className="px-4 py-2 text-sm text-gray-300 hover:bg-gray-700">
                    Configure Models
                  </div>
                </div>
              )}
            </div>
            
            {/* User menu */}
            <div className="relative">
              <button
                onClick={handleUserMenuToggle}
                className="flex text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <div className="h-8 w-8 rounded-full bg-blue-700 border-2 border-blue-400/20 flex items-center justify-center text-white">
                  U
                </div>
              </button>
              
              {/* User dropdown menu */}
              {userMenuOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-gray-800 border border-gray-700 rounded-lg shadow-lg py-1 z-10">
                  <div className="px-4 py-2 border-b border-gray-700">
                    <p className="text-sm font-medium text-white">User</p>
                    <p className="text-xs text-gray-400">user@example.com</p>
                  </div>
                  <div className="px-4 py-2 text-sm text-gray-300 hover:bg-gray-700">
                    Settings
                  </div>
                  <div className="px-4 py-2 text-sm text-gray-300 hover:bg-gray-700">
                    About
                  </div>
                  <div className="border-t border-gray-700 mt-1"></div>
                  <div className="px-4 py-2 text-sm text-red-400 hover:bg-gray-700">
                    Logout
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-gray-900 border-t border-gray-800">
          <div className="px-2 pt-2 pb-3 space-y-1">
            {navItems.map((item) => (
              <button
                key={item.name}
                onClick={() => handleNavigation(item.path)}
                className={`block w-full text-left px-3 py-2 rounded-md text-base font-medium ${
                  isActive(item.path)
                    ? 'bg-blue-900/30 text-blue-400'
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                }`}
              >
                {item.name}
              </button>
            ))}
          </div>
        </div>
      )}
    </header>
  );
}

export default Header; 