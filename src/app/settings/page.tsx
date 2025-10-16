'use client';

import { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { UserIcon, CreditCardIcon, KeyIcon, BellIcon } from '@heroicons/react/24/outline';

export default function SettingsPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState('profile');

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

  const tabs = [
    { id: 'profile', name: 'Profile', icon: UserIcon },
    { id: 'billing', name: 'Billing', icon: CreditCardIcon },
    { id: 'api', name: 'API Keys', icon: KeyIcon },
    { id: 'notifications', name: 'Notifications', icon: BellIcon }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-8">Settings</h1>

        <div className="grid md:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="md:col-span-1">
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-4">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center px-4 py-3 rounded-lg mb-2 transition-all ${
                    activeTab === tab.id
                      ? 'bg-purple-600/30 text-white'
                      : 'text-gray-400 hover:bg-white/5'
                  }`}
                >
                  <tab.icon className="h-5 w-5 mr-3" />
                  {tab.name}
                </button>
              ))}
            </div>
          </div>

          {/* Content */}
          <div className="md:col-span-3">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6"
            >
              {activeTab === 'profile' && (
                <div>
                  <h2 className="text-2xl font-bold text-white mb-6">Profile Settings</h2>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Name</label>
                      <input
                        type="text"
                        className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white"
                        defaultValue="John Doe"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
                      <input
                        type="email"
                        className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white"
                        defaultValue="john@example.com"
                      />
                    </div>
                    <button className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700">
                      Save Changes
                    </button>
                  </div>
                </div>
              )}

              {activeTab === 'billing' && (
                <div>
                  <h2 className="text-2xl font-bold text-white mb-6">Billing & Subscription</h2>
                  <div className="space-y-6">
                    <div className="bg-purple-600/20 border border-purple-500/50 rounded-lg p-4">
                      <div className="flex justify-between items-center">
                        <div>
                          <h3 className="text-lg font-semibold text-white">Pro Plan</h3>
                          <p className="text-gray-400">$29/month</p>
                        </div>
                        <button className="px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20">
                          Upgrade
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'api' && (
                <div>
                  <h2 className="text-2xl font-bold text-white mb-6">API Keys</h2>
                  <button className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 mb-4">
                    Generate New Key
                  </button>
                  <div className="text-gray-400">No API keys yet</div>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
