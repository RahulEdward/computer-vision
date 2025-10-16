'use client';

import { useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { UsersIcon, ServerIcon, ChartBarIcon, CogIcon } from '@heroicons/react/24/outline';

export default function AdminDashboard() {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/auth/login');
    }
  }, [status, router]);

  if (status === 'loading') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  if (!session) return null;
  const stats = [
    { name: 'Total Users', value: '1,234', icon: UsersIcon, change: '+12%' },
    { name: 'Active Workspaces', value: '456', icon: ServerIcon, change: '+8%' },
    { name: 'Total Executions', value: '89.2K', icon: ChartBarIcon, change: '+23%' },
    { name: 'System Health', value: '99.9%', icon: CogIcon, change: '+0.1%' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-8">Admin Dashboard</h1>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <stat.icon className="h-8 w-8 text-purple-400" />
                <span className="text-green-400 text-sm font-medium">{stat.change}</span>
              </div>
              <h3 className="text-3xl font-bold text-white mb-1">{stat.value}</h3>
              <p className="text-gray-400 text-sm">{stat.name}</p>
            </motion.div>
          ))}
        </div>

        {/* Recent Activity */}
        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
          <h2 className="text-xl font-bold text-white mb-4">Recent Activity</h2>
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="flex items-center justify-between py-3 border-b border-white/10">
                <div>
                  <p className="text-white">User signed up</p>
                  <p className="text-gray-400 text-sm">2 minutes ago</p>
                </div>
                <span className="px-3 py-1 bg-green-600/20 text-green-400 rounded-full text-sm">
                  Success
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
