'use client';

import { useState, useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';

export default function OnboardingPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [step, setStep] = useState(1);

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

  const [data, setData] = useState({
    workspaceName: '',
    role: '',
    useCase: ''
  });

  const handleComplete = () => {
    // TODO: Save onboarding data
    router.push('/dashboard');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-2xl"
      >
        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-8">
          {/* Progress */}
          <div className="mb-8">
            <div className="flex justify-between mb-2">
              {[1, 2, 3].map((s) => (
                <div
                  key={s}
                  className={`w-1/3 h-2 rounded-full mx-1 ${s <= step ? 'bg-purple-600' : 'bg-white/10'
                    }`}
                />
              ))}
            </div>
            <p className="text-gray-400 text-sm">Step {step} of 3</p>
          </div>

          {/* Step 1 */}
          {step === 1 && (
            <div>
              <h2 className="text-3xl font-bold text-white mb-4">Welcome to Computer Genie! 🧞‍♂️</h2>
              <p className="text-gray-400 mb-6">Let's set up your workspace</p>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Workspace Name
                  </label>
                  <input
                    type="text"
                    value={data.workspaceName}
                    onChange={(e) => setData({ ...data, workspaceName: e.target.value })}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white"
                    placeholder="My Company"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Step 2 */}
          {step === 2 && (
            <div>
              <h2 className="text-3xl font-bold text-white mb-4">What's your role?</h2>
              <div className="grid grid-cols-2 gap-4">
                {['Developer', 'Designer', 'Manager', 'Other'].map((role) => (
                  <button
                    key={role}
                    onClick={() => setData({ ...data, role })}
                    className={`p-4 rounded-lg border-2 transition-all ${data.role === role
                      ? 'border-purple-500 bg-purple-600/20'
                      : 'border-white/10 bg-white/5 hover:bg-white/10'
                      }`}
                  >
                    <span className="text-white font-medium">{role}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step 3 */}
          {step === 3 && (
            <div>
              <h2 className="text-3xl font-bold text-white mb-4">What will you automate?</h2>
              <div className="space-y-3">
                {['Data Processing', 'API Integration', 'File Management', 'Other'].map((useCase) => (
                  <button
                    key={useCase}
                    onClick={() => setData({ ...data, useCase })}
                    className={`w-full p-4 rounded-lg border-2 text-left transition-all ${data.useCase === useCase
                      ? 'border-purple-500 bg-purple-600/20'
                      : 'border-white/10 bg-white/5 hover:bg-white/10'
                      }`}
                  >
                    <span className="text-white font-medium">{useCase}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Navigation */}
          <div className="flex justify-between mt-8">
            {step > 1 && (
              <button
                onClick={() => setStep(step - 1)}
                className="px-6 py-2 bg-white/10 text-white rounded-lg hover:bg-white/20"
              >
                Back
              </button>
            )}
            <button
              onClick={() => (step === 3 ? handleComplete() : setStep(step + 1))}
              className="ml-auto px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
            >
              {step === 3 ? 'Get Started' : 'Continue'}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
