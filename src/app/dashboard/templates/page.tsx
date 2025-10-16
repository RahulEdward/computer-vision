'use client';

import { useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import DashboardHeader from '@/components/layout/DashboardHeader';
import { motion } from 'framer-motion';
import {
  RocketLaunchIcon,
  EnvelopeIcon,
  ShoppingCartIcon,
  ChartBarIcon,
  CloudIcon,
  BellIcon,
} from '@heroicons/react/24/outline';

export default function TemplatesPage() {
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

  const templates = [
    {
      id: 1,
      name: 'Email Automation',
      description: 'Automatically send emails based on triggers',
      icon: EnvelopeIcon,
      color: 'from-blue-500 to-blue-600',
      nodes: 5,
    },
    {
      id: 2,
      name: 'E-commerce Order Processing',
      description: 'Process orders and update inventory',
      icon: ShoppingCartIcon,
      color: 'from-green-500 to-green-600',
      nodes: 8,
    },
    {
      id: 3,
      name: 'Data Analytics Pipeline',
      description: 'Collect, process, and analyze data',
      icon: ChartBarIcon,
      color: 'from-purple-500 to-purple-600',
      nodes: 6,
    },
    {
      id: 4,
      name: 'Social Media Posting',
      description: 'Auto-post to multiple platforms',
      icon: RocketLaunchIcon,
      color: 'from-pink-500 to-pink-600',
      nodes: 7,
    },
    {
      id: 5,
      name: 'Cloud Backup',
      description: 'Automated backup to cloud storage',
      icon: CloudIcon,
      color: 'from-indigo-500 to-indigo-600',
      nodes: 4,
    },
    {
      id: 6,
      name: 'Notification System',
      description: 'Send alerts via multiple channels',
      icon: BellIcon,
      color: 'from-orange-500 to-orange-600',
      nodes: 5,
    },
  ];

  return (
    <div className="min-h-screen bg-[#0a0118]">
      <DashboardHeader />
      
      <div className="max-w-7xl mx-auto px-6 py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-white mb-2">Workflow Templates</h1>
          <p className="text-gray-400">Pre-built templates to get started quickly</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {templates.map((template, index) => (
            <motion.div
              key={template.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Link href="/dashboard/workflows">
                <div className="bg-[#1a1625] border border-white/10 rounded-xl p-6 hover:border-purple-500/50 transition-all cursor-pointer group">
                  <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${template.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                    <template.icon className="h-6 w-6 text-white" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{template.name}</h3>
                  <p className="text-gray-400 text-sm mb-4">{template.description}</p>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500">{template.nodes} nodes</span>
                    <span className="text-purple-400 group-hover:text-purple-300">Use Template →</span>
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
