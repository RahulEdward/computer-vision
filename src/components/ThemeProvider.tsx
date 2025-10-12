'use client';

import { useEffect } from 'react';
import { useDashboardStore } from '@/lib/store';

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const { theme, setTheme } = useDashboardStore();

  // Initialize theme from localStorage on mount
  useEffect(() => {
    console.log('🎨 ThemeProvider: Initializing theme...');
    const savedTheme = localStorage.getItem('theme');
    console.log('💾 Saved theme from localStorage:', savedTheme);
    
    if (savedTheme === 'light' || savedTheme === 'dark') {
      console.log('✅ Using saved theme:', savedTheme);
      setTheme(savedTheme);
    } else {
      // Check system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const defaultTheme = prefersDark ? 'dark' : 'light';
      console.log('🌐 Using system preference:', defaultTheme);
      setTheme(defaultTheme);
    }
  }, [setTheme]);

  // Apply theme class to document (backup in case store doesn't do it)
  useEffect(() => {
    console.log('🎨 ThemeProvider: Theme changed to', theme);
    const root = document.documentElement;
    const body = document.body;
    
    if (theme === 'dark') {
      root.classList.add('dark');
      body.classList.add('dark');
      console.log('✅ ThemeProvider: Applied dark classes');
    } else {
      root.classList.remove('dark');
      body.classList.remove('dark');
      console.log('✅ ThemeProvider: Removed dark classes');
    }
    
    console.log('📍 Current classes - HTML:', root.className, 'Body:', body.className);
  }, [theme]);

  return <>{children}</>;
}
