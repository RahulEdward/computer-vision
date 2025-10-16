'use client';

import { useState } from 'react';
import { useSession } from 'next-auth/react';
import { motion } from 'framer-motion';
import { UserIcon, CreditCardIcon, KeyIcon, BellIcon, ClipboardIcon, TrashIcon } from '@heroicons/react/24/outline';
import DashboardHeader from '@/components/layout/DashboardHeader';

interface ApiKey {
  id: string;
  name: string;
  key: string;
  createdAt: string;
  lastUsed?: string;
}

export default function SettingsPage() {
  const { data: session } = useSession();
  const [activeTab, setActiveTab] = useState('profile');
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [showNewKeyModal, setShowNewKeyModal] = useState(false);
  const [newKeyName, setNewKeyName] = useState('');
  const [generatedKey, setGeneratedKey] = useState('');

  const generateApiKey = () => {
    if (!newKeyName.trim()) {
      alert('Please enter a name for the API key');
      return;
    }

    // Generate a random API key
    const key = 'cg_' + Array.from({ length: 32 }, () => 
      Math.random().toString(36).charAt(2)
    ).join('');

    const newKey: ApiKey = {
      id: Date.now().toString(),
      name: newKeyName,
      key: key,
      createdAt: new Date().toISOString(),
    };

    setApiKeys([...apiKeys, newKey]);
    setGeneratedKey(key);
    setNewKeyName('');
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    alert('API key copied to clipboard!');
  };

  const deleteApiKey = (id: string) => {
    if (confirm('Are you sure you want to delete this API key? This action cannot be undone.')) {
      setApiKeys(apiKeys.filter(key => key.id !== id));
    }
  };

  const closeModal = () => {
    setShowNewKeyModal(false);
    setGeneratedKey('');
    setNewKeyName('');
  };

  const tabs = [
    { id: 'profile', name: 'Profile', icon: UserIcon },
    { id: 'billing', name: 'Billing', icon: CreditCardIcon },
    { id: 'api', name: 'API Keys', icon: KeyIcon },
    { id: 'notifications', name: 'Notifications', icon: BellIcon }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <DashboardHeader />
      
      <div className="p-6">
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
                          className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                          defaultValue={session?.user?.name || 'John Doe'}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Email</label>
                        <input
                          type="email"
                          className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                          defaultValue={session?.user?.email || 'john@example.com'}
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Company</label>
                        <input
                          type="text"
                          className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                          placeholder="Your company name"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">Bio</label>
                        <textarea
                          className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                          rows={4}
                          placeholder="Tell us about yourself"
                        />
                      </div>
                      <button className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                        Save Changes
                      </button>
                    </div>
                  </div>
                )}

                {activeTab === 'billing' && (
                  <div>
                    <h2 className="text-2xl font-bold text-white mb-6">Billing & Subscription</h2>
                    <div className="space-y-6">
                      {/* Current Plan */}
                      <div className="bg-purple-600/20 border border-purple-500/50 rounded-lg p-6">
                        <div className="flex justify-between items-center mb-4">
                          <div>
                            <h3 className="text-lg font-semibold text-white">Free Plan</h3>
                            <p className="text-gray-400">$0/month</p>
                          </div>
                          <span className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm">
                            Active
                          </span>
                        </div>
                        <div className="space-y-2 text-sm text-gray-300">
                          <p>✓ 5 workflows</p>
                          <p>✓ 100 executions/month</p>
                          <p>✓ Basic templates</p>
                        </div>
                      </div>

                      {/* Upgrade Options */}
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="bg-white/5 border border-white/10 rounded-lg p-6">
                          <h3 className="text-lg font-semibold text-white mb-2">Pro Plan</h3>
                          <p className="text-3xl font-bold text-white mb-4">$29<span className="text-sm text-gray-400">/month</span></p>
                          <div className="space-y-2 text-sm text-gray-300 mb-4">
                            <p>✓ Unlimited workflows</p>
                            <p>✓ 10,000 executions/month</p>
                            <p>✓ All templates</p>
                            <p>✓ Priority support</p>
                          </div>
                          <button className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                            Upgrade to Pro
                          </button>
                        </div>

                        <div className="bg-white/5 border border-white/10 rounded-lg p-6">
                          <h3 className="text-lg font-semibold text-white mb-2">Enterprise</h3>
                          <p className="text-3xl font-bold text-white mb-4">$99<span className="text-sm text-gray-400">/month</span></p>
                          <div className="space-y-2 text-sm text-gray-300 mb-4">
                            <p>✓ Everything in Pro</p>
                            <p>✓ Unlimited executions</p>
                            <p>✓ Custom integrations</p>
                            <p>✓ Dedicated support</p>
                          </div>
                          <button className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                            Upgrade to Enterprise
                          </button>
                        </div>
                      </div>

                      {/* Payment Method */}
                      <div className="bg-white/5 border border-white/10 rounded-lg p-6">
                        <h3 className="text-lg font-semibold text-white mb-4">Payment Method</h3>
                        <p className="text-gray-400 text-sm mb-4">No payment method added</p>
                        <button className="px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors">
                          Add Payment Method
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'api' && (
                  <div>
                    <h2 className="text-2xl font-bold text-white mb-6">API Keys</h2>
                    <p className="text-gray-400 mb-6">
                      API keys allow you to integrate Computer Genie with your applications.
                    </p>
                    <button 
                      onClick={() => setShowNewKeyModal(true)}
                      className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 mb-6 transition-colors"
                    >
                      Generate New Key
                    </button>

                    {/* API Keys List */}
                    {apiKeys.length === 0 ? (
                      <div className="bg-white/5 border border-white/10 rounded-lg p-6">
                        <p className="text-gray-400 text-center">No API keys yet</p>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {apiKeys.map((apiKey) => (
                          <div key={apiKey.id} className="bg-white/5 border border-white/10 rounded-lg p-4">
                            <div className="flex items-center justify-between mb-2">
                              <h3 className="text-white font-medium">{apiKey.name}</h3>
                              <button
                                onClick={() => deleteApiKey(apiKey.id)}
                                className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-lg transition-colors"
                                title="Delete API Key"
                              >
                                <TrashIcon className="h-5 w-5" />
                              </button>
                            </div>
                            <div className="flex items-center space-x-2 mb-2">
                              <code className="flex-1 px-3 py-2 bg-black/40 text-gray-300 rounded text-sm font-mono">
                                {apiKey.key.substring(0, 20)}...{apiKey.key.substring(apiKey.key.length - 4)}
                              </code>
                              <button
                                onClick={() => copyToClipboard(apiKey.key)}
                                className="p-2 text-purple-400 hover:text-purple-300 hover:bg-purple-500/10 rounded-lg transition-colors"
                                title="Copy to clipboard"
                              >
                                <ClipboardIcon className="h-5 w-5" />
                              </button>
                            </div>
                            <div className="flex items-center space-x-4 text-xs text-gray-400">
                              <span>Created: {new Date(apiKey.createdAt).toLocaleDateString()}</span>
                              {apiKey.lastUsed && <span>Last used: {new Date(apiKey.lastUsed).toLocaleDateString()}</span>}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* New API Key Modal */}
                    {showNewKeyModal && (
                      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
                        <motion.div
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="bg-[#2a2435] border border-white/10 rounded-xl p-6 max-w-md w-full mx-4"
                        >
                          {!generatedKey ? (
                            <>
                              <h3 className="text-xl font-bold text-white mb-4">Generate New API Key</h3>
                              <p className="text-gray-400 text-sm mb-4">
                                Give your API key a name to help you identify it later.
                              </p>
                              <input
                                type="text"
                                value={newKeyName}
                                onChange={(e) => setNewKeyName(e.target.value)}
                                placeholder="e.g., Production API, Development Key"
                                className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 mb-4"
                              />
                              <div className="flex space-x-3">
                                <button
                                  onClick={closeModal}
                                  className="flex-1 px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
                                >
                                  Cancel
                                </button>
                                <button
                                  onClick={generateApiKey}
                                  className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                                >
                                  Generate
                                </button>
                              </div>
                            </>
                          ) : (
                            <>
                              <h3 className="text-xl font-bold text-white mb-4">API Key Generated!</h3>
                              <p className="text-gray-400 text-sm mb-4">
                                Copy this key now. You won't be able to see it again!
                              </p>
                              <div className="bg-black/40 border border-white/20 rounded-lg p-4 mb-4">
                                <code className="text-green-400 text-sm font-mono break-all">
                                  {generatedKey}
                                </code>
                              </div>
                              <div className="flex space-x-3">
                                <button
                                  onClick={() => copyToClipboard(generatedKey)}
                                  className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                                >
                                  Copy Key
                                </button>
                                <button
                                  onClick={closeModal}
                                  className="flex-1 px-4 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20 transition-colors"
                                >
                                  Done
                                </button>
                              </div>
                            </>
                          )}
                        </motion.div>
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'notifications' && (
                  <div>
                    <h2 className="text-2xl font-bold text-white mb-6">Notification Settings</h2>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-4 bg-white/5 border border-white/10 rounded-lg">
                        <div>
                          <h3 className="text-white font-medium">Email Notifications</h3>
                          <p className="text-gray-400 text-sm">Receive email updates about your workflows</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input type="checkbox" className="sr-only peer" defaultChecked />
                          <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                        </label>
                      </div>

                      <div className="flex items-center justify-between p-4 bg-white/5 border border-white/10 rounded-lg">
                        <div>
                          <h3 className="text-white font-medium">Workflow Alerts</h3>
                          <p className="text-gray-400 text-sm">Get notified when workflows fail</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input type="checkbox" className="sr-only peer" defaultChecked />
                          <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                        </label>
                      </div>

                      <div className="flex items-center justify-between p-4 bg-white/5 border border-white/10 rounded-lg">
                        <div>
                          <h3 className="text-white font-medium">Usage Alerts</h3>
                          <p className="text-gray-400 text-sm">Alert when approaching usage limits</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input type="checkbox" className="sr-only peer" />
                          <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                        </label>
                      </div>

                      <div className="flex items-center justify-between p-4 bg-white/5 border border-white/10 rounded-lg">
                        <div>
                          <h3 className="text-white font-medium">Marketing Emails</h3>
                          <p className="text-gray-400 text-sm">Receive updates about new features</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input type="checkbox" className="sr-only peer" />
                          <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-800 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                        </label>
                      </div>
                    </div>
                  </div>
                )}
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
