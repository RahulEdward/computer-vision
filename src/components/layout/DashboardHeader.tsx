'use client';

import { useSession, signOut } from 'next-auth/react';
import Link from 'next/link';
import { useState } from 'react';
import {
  UserCircleIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon,
  ChevronDownIcon,
  BellIcon,
  PlusIcon
} from '@heroicons/react/24/outline';

export default function DashboardHeader() {
  const { data: session } = useSession();
  const [showUserMenu, setShowUserMenu] = useState(false);

  return (
    <header className="bg-[#1a1625] border-b border-white/10 px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Logo and Title */}
        <div className="flex items-center space-x-4">
          <Link href="/dashboard" className="flex items-center space-x-2">
            <span className="text-2xl">🧞‍♂️</span>
            <span className="text-xl font-bold text-white">Computer Genie</span>
          </Link>
          
          <div className="h-6 w-px bg-white/20"></div>
          
          <nav className="flex items-center space-x-1">
            <Link 
              href="/dashboard" 
              className="px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-white/5 rounded-lg"
            >
              Dashboard
            </Link>
            <Link 
              href="/dashboard/workflows" 
              className="px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-white/5 rounded-lg"
            >
              Workflows
            </Link>
            <Link 
              href="/dashboard/executions" 
              className="px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-white/5 rounded-lg"
            >
              Executions
            </Link>
            <Link 
              href="/dashboard/templates" 
              className="px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-white/5 rounded-lg"
            >
              Templates
            </Link>
          </nav>
        </div>

        {/* Right Side Actions */}
        <div className="flex items-center space-x-3">
          {/* New Workflow Button */}
          <Link href="/dashboard/workflows">
            <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 text-sm font-medium">
              <PlusIcon className="h-4 w-4" />
              <span>New Workflow</span>
            </button>
          </Link>

          {/* Notifications */}
          <button className="p-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-lg relative">
            <BellIcon className="h-5 w-5" />
            <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-full"></span>
          </button>

          {/* Settings */}
          <Link 
            href="/dashboard/settings" 
            className="p-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-lg"
          >
            <Cog6ToothIcon className="h-5 w-5" />
          </Link>

          {/* User Menu */}
          <div className="relative">
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center space-x-2 px-3 py-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-lg"
            >
              <UserCircleIcon className="h-6 w-6" />
              <span className="text-sm">{session?.user?.name || 'User'}</span>
              <ChevronDownIcon className="h-4 w-4" />
            </button>

            {showUserMenu && (
              <div className="absolute right-0 mt-2 w-48 bg-[#2a2435] border border-white/10 rounded-lg shadow-xl py-1 z-50">
                <div className="px-4 py-2 border-b border-white/10">
                  <p className="text-sm text-white font-medium">{session?.user?.name}</p>
                  <p className="text-xs text-gray-400">{session?.user?.email}</p>
                </div>
                
                <Link
                  href="/dashboard/settings"
                  className="flex items-center space-x-2 px-4 py-2 text-sm text-gray-300 hover:text-white hover:bg-white/5"
                >
                  <Cog6ToothIcon className="h-4 w-4" />
                  <span>Settings</span>
                </Link>
                
                <button
                  onClick={() => signOut({ callbackUrl: '/' })}
                  className="flex items-center space-x-2 w-full px-4 py-2 text-sm text-red-400 hover:text-red-300 hover:bg-white/5"
                >
                  <ArrowRightOnRectangleIcon className="h-4 w-4" />
                  <span>Sign Out</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
