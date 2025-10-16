'use client';

import { useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import DashboardHeader from '@/components/layout/DashboardHeader';
import { motion } from 'framer-motion';
import {
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  PlayIcon,
} from '@heroicons/react/24/outline';

export default function ExecutionsPage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/auth/login');
    }
  }, [status, router]);

  if (status === 'loading') {
    return (
      <div className="min-h-screen bg-[#0a0118] flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  if (!session) {
    return null;
  }

  const executions = [
    { id: 1, workflow: 'Email Automation', status: 'success', duration: '2.3s', time: '2 mins ago' },
    { id: 2, workflow: 'Data Processing', status: 'success', duration: '5.1s', time: '15 mins ago' },
    { id: 3, workflow: 'API Integration', status: 'running', duration: '-', time: 'Now' },
    { id: 4, workflow: 'Backup Task', status: 'failed', duration: '1.2s', time: '1 hour ago' },
    { id: 5, workflow: 'Social Media Post', status: 'success', duration: '3.4s', time: '2 hours ago' },
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircleIcon className="h-5 w-5 text-green-400" />;
      case 'failed':
        return <XCircleIcon className="h-5 w-5 text-red-400" />;
      case 'running':
        return <PlayIcon className="h-5 w-5 text-blue-400 animate-pulse" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'bg-green-500/20 text-green-400';
      case 'failed':
        return 'bg-red-500/20 text-red-400';
      case 'running':
        return 'bg-blue-500/20 text-blue-400';
      default:
        return 'bg-gray-500/20 text-gray-400';
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0118]">
      <DashboardHeader />
      
      <div className="max-w-7xl mx-auto px-6 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-white mb-2">Workflow Executions</h1>
          <p className="text-gray-400">Monitor your workflow runs and performance</p>
        </motion.div>

        <div className="bg-[#1a1625] border border-white/10 rounded-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-white/5 border-b border-white/10">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Workflow</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Duration</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Time</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/10">
                {executions.map((execution, index) => (
                  <motion.tr
                    key={execution.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="hover:bg-white/5 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(execution.status)}
                        <span className="text-white font-medium">{execution.workflow}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(execution.status)}`}>
                        {execution.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-400">{execution.duration}</td>
                    <td className="px-6 py-4 text-gray-400">{execution.time}</td>
                    <td className="px-6 py-4">
                      <button className="text-purple-400 hover:text-purple-300 text-sm">
                        View Details →
                      </button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
