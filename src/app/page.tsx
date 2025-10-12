'use client';

import dynamic from 'next/dynamic';
import { Suspense } from 'react';

// Dynamically import the main dashboard to avoid SSR issues
const MainDashboard = dynamic(() => import('@/components/dashboard/MainDashboard'), {
  ssr: false,
  loading: () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 mx-auto mb-4 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <h2 className="text-xl font-semibold text-slate-700 dark:text-slate-300 mb-2">
          Loading Computer Genie Dashboard
        </h2>
        <p className="text-slate-600 dark:text-slate-400">
          Initializing advanced features...
        </p>
      </div>
    </div>
  )
});

export default function Home() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <h2 className="text-xl font-semibold text-slate-700 dark:text-slate-300 mb-2">
            Loading Computer Genie Dashboard
          </h2>
          <p className="text-slate-600 dark:text-slate-400">
            Preparing your automation workspace...
          </p>
        </div>
      </div>
    }>
      <MainDashboard />
    </Suspense>
  );
}
