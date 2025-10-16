'use client';

import { useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import DashboardHeader from '@/components/layout/DashboardHeader';
import {
  RocketLaunchIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  PlusIcon,
  ChartBarIcon,
  CpuChipIcon,
  CloudIcon
} from '@heroicons/react/24/outline';

export default function DashboardPage() {
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

  return (
    <div className="min-h-screen bg-[#0a0118]">
      <DashboardHeader />
      
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-white mb-2">
            Welcome back, {session.user?.name}! 👋
          </h1>
          <p className="text-gray-400">
            Here's what's happening with your workflows today
          </p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-[#1a1625] border border-white/10 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <RocketLaunchIcon className="h-8 w-8 text-purple-400" />
              <span className="text-2xl font-bold text-white">12</span>
            </div>
            <p className="text-gray-400 text-sm">Active Workflows</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-[#1a1625] border border-white/10 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <CheckCircleIcon className="h-8 w-8 text-green-400" />
              <span className="text-2xl font-bold text-white">1,247</span>
            </div>
            <p className="text-gray-400 text-sm">Successful Runs</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-[#1a1625] border border-white/10 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <ClockIcon className="h-8 w-8 text-blue-400" />
              <span className="text-2xl font-bold text-white">3</span>
            </div>
            <p className="text-gray-400 text-sm">Running Now</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-[#1a1625] border border-white/10 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <XCircleIcon className="h-8 w-8 text-red-400" />
              <span className="text-2xl font-bold text-white">2</span>
            </div>
            <p className="text-gray-400 text-sm">Failed Today</p>
          </motion.div>
        </div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mb-8"
        >
          <h2 className="text-xl font-semibold text-white mb-4">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Link
              href="/dashboard/workflows"
              className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl p-6 hover:from-purple-700 hover:to-pink-700 transition-all"
            >
              <PlusIcon className="h-8 w-8 text-white mb-3" />
              <h3 className="text-lg font-semibold text-white mb-1">Create Workflow</h3>
              <p className="text-white/80 text-sm">Build a new automation workflow</p>
            </Link>

            <Link
              href="/dashboard/templates"
              className="bg-[#1a1625] border border-white/10 rounded-xl p-6 hover:border-purple-500/50 transition-all"
            >
              <CpuChipIcon className="h-8 w-8 text-purple-400 mb-3" />
              <h3 className="text-lg font-semibold text-white mb-1">Browse Templates</h3>
              <p className="text-gray-400 text-sm">Start from pre-built templates</p>
            </Link>

            <Link
              href="/dashboard/executions"
              className="bg-[#1a1625] border border-white/10 rounded-xl p-6 hover:border-purple-500/50 transition-all"
            >
              <ChartBarIcon className="h-8 w-8 text-blue-400 mb-3" />
              <h3 className="text-lg font-semibold text-white mb-1">View Analytics</h3>
              <p className="text-gray-400 text-sm">Monitor workflow performance</p>
            </Link>
          </div>
        </motion.div>

        {/* Recent Workflows */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-white">Recent Workflows</h2>
            <Link href="/dashboard/workflows" className="text-purple-400 hover:text-purple-300 text-sm">
              View All →
            </Link>
          </div>
          
          <div className="bg-[#1a1625] border border-white/10 rounded-xl overflow-hidden">
            <div className="divide-y divide-white/10">
              {[
                { name: 'My Automation Workflow', status: 'active', runs: 247, lastRun: '2 mins ago' },
                { name: 'Data Processing Pipeline', status: 'active', runs: 156, lastRun: '15 mins ago' },
                { name: 'Email Notification System', status: 'paused', runs: 89, lastRun: '1 hour ago' },
              ].map((workflow, index) => (
                <div key={index} className="p-4 hover:bg-white/5 transition-colors">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="h-10 w-10 bg-purple-600/20 rounded-lg flex items-center justify-center">
                        <CloudIcon className="h-6 w-6 text-purple-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">{workflow.name}</h3>
                        <p className="text-gray-400 text-sm">
                          {workflow.runs} runs • Last run {workflow.lastRun}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        workflow.status === 'active' 
                          ? 'bg-green-500/20 text-green-400' 
                          : 'bg-gray-500/20 text-gray-400'
                      }`}>
                        {workflow.status}
                      </span>
                      <Link
                        href="/dashboard/workflows"
                        className="px-4 py-2 bg-purple-600/20 text-purple-400 rounded-lg hover:bg-purple-600/30 text-sm"
                      >
                        Open
                      </Link>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
