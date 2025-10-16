'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { 
  BoltIcon, 
  CloudIcon, 
  ShieldCheckIcon, 
  ChartBarIcon,
  UsersIcon,
  CogIcon,
  SparklesIcon,
  RocketLaunchIcon,
  CheckCircleIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline';

export default function LandingPage() {
  const features = [
    {
      icon: BoltIcon,
      title: 'Lightning Fast',
      description: 'Execute workflows in milliseconds with our optimized engine',
      color: 'from-yellow-400 to-orange-500'
    },
    {
      icon: CloudIcon,
      title: 'Cloud Native',
      description: 'Built for scale with modern cloud infrastructure',
      color: 'from-blue-400 to-cyan-500'
    },
    {
      icon: ShieldCheckIcon,
      title: 'Enterprise Security',
      description: 'Bank-grade encryption and compliance certifications',
      color: 'from-green-400 to-emerald-500'
    },
    {
      icon: ChartBarIcon,
      title: 'Advanced Analytics',
      description: 'Real-time insights into your automation performance',
      color: 'from-purple-400 to-pink-500'
    },
    {
      icon: UsersIcon,
      title: 'Team Collaboration',
      description: 'Work together seamlessly with your team',
      color: 'from-indigo-400 to-purple-500'
    },
    {
      icon: CogIcon,
      title: 'Easy Integration',
      description: 'Connect with 1000+ apps and services',
      color: 'from-pink-400 to-rose-500'
    }
  ];

  return (
    <div className="min-h-screen bg-[#0a0118] relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse delay-500"></div>
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-black/50 backdrop-blur-xl border-b border-white/5 z-50">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3"
            >
              <div className="text-4xl">🧞‍♂️</div>
              <span className="text-2xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent">
                Computer Genie
              </span>
            </motion.div>
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-8"
            >
              <Link href="/pricing" className="text-gray-300 hover:text-white transition-colors font-medium">
                Pricing
              </Link>
              <Link href="/auth/login" className="text-gray-300 hover:text-white transition-colors font-medium">
                Login
              </Link>
              <Link href="/auth/signup">
                <button className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:from-purple-700 hover:to-pink-700 transition-all transform hover:scale-105 font-semibold shadow-lg shadow-purple-500/30">
                  Get Started Free
                </button>
              </Link>
            </motion.div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center max-w-5xl mx-auto">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center gap-2 bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-full px-6 py-3 mb-8 backdrop-blur-sm"
            >
              <SparklesIcon className="w-5 h-5 text-purple-400" />
              <span className="text-purple-300 font-medium">Trusted by 10,000+ teams worldwide</span>
            </motion.div>

            {/* Main Headline */}
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="text-7xl md:text-8xl lg:text-9xl font-black mb-8 leading-none"
            >
              <span className="text-white block mb-4">Automate</span>
              <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-purple-400 bg-clip-text text-transparent bg-[length:200%_auto] animate-gradient">
                Anything
              </span>
            </motion.h1>

            {/* Subheadline */}
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="text-xl md:text-2xl text-gray-300 mb-12 leading-relaxed max-w-3xl mx-auto"
            >
              The most powerful workflow automation platform.
              <br />
              <span className="text-purple-400 font-semibold">Visual programming meets AI</span> to help you build complex automations without code.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12"
            >
              <Link href="/auth/signup">
                <button className="group px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-2xl text-lg font-bold hover:from-purple-700 hover:to-pink-700 transition-all shadow-2xl shadow-purple-500/50 flex items-center gap-3 hover:gap-4">
                  <RocketLaunchIcon className="w-6 h-6" />
                  <span>Start Free Trial</span>
                  <ArrowRightIcon className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </button>
              </Link>
              <Link href="/pricing">
                <button className="px-8 py-4 bg-white/5 backdrop-blur-xl border-2 border-white/10 text-white rounded-2xl text-lg font-bold hover:bg-white/10 transition-all">
                  View Pricing
                </button>
              </Link>
            </motion.div>

            {/* Trust Badges */}
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="flex flex-wrap items-center justify-center gap-6 text-gray-400 text-sm"
            >
              <div className="flex items-center gap-2">
                <CheckCircleIcon className="w-5 h-5 text-green-400" />
                <span>No credit card required</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircleIcon className="w-5 h-5 text-green-400" />
                <span>14-day free trial</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircleIcon className="w-5 h-5 text-green-400" />
                <span>Cancel anytime</span>
              </div>
            </motion.div>
          </div>

          {/* Demo Preview */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="mt-20 relative max-w-6xl mx-auto"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-600/30 to-pink-600/30 blur-3xl"></div>
            <div className="relative bg-gradient-to-br from-slate-900/80 to-purple-900/80 backdrop-blur-2xl border border-white/10 rounded-3xl p-2 shadow-2xl">
              <div className="bg-gradient-to-br from-purple-500/5 to-pink-500/5 rounded-2xl p-8 aspect-video flex items-center justify-center">
                <div className="text-center">
                  <div className="text-6xl mb-4">🎬</div>
                  <p className="text-2xl text-white font-semibold mb-2">Product Demo</p>
                  <p className="text-gray-400">See Computer Genie in action</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { number: '10K+', label: 'Active Users', color: 'from-blue-400 to-cyan-400' },
              { number: '1M+', label: 'Workflows Executed', color: 'from-purple-400 to-pink-400' },
              { number: '99.9%', label: 'Uptime', color: 'from-green-400 to-emerald-400' }
            ].map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="text-center p-8 bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl"
              >
                <div className={`text-6xl font-black bg-gradient-to-r ${stat.color} bg-clip-text text-transparent mb-4`}>
                  {stat.number}
                </div>
                <div className="text-lg text-gray-400 font-medium">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl md:text-6xl font-black text-white mb-6">
              Everything You Need
            </h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              Powerful features to supercharge your automation workflow
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="group relative bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 hover:bg-white/10 hover:border-white/20 transition-all cursor-pointer"
              >
                <div className={`w-14 h-14 bg-gradient-to-r ${feature.color} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                  <feature.icon className="h-7 w-7 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-400 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative py-32 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 rounded-3xl blur-2xl opacity-30"></div>
            <div className="relative bg-gradient-to-r from-purple-600 to-pink-600 rounded-3xl p-16 text-center">
              <h2 className="text-5xl md:text-6xl font-black text-white mb-6">
                Ready to Get Started?
              </h2>
              <p className="text-xl text-white/90 mb-10 max-w-2xl mx-auto">
                Join thousands of teams automating their workflows with Computer Genie
              </p>
              <Link href="/auth/signup">
                <button className="group px-10 py-5 bg-white text-purple-600 rounded-2xl text-lg font-bold hover:bg-gray-100 transition-all shadow-2xl inline-flex items-center gap-3">
                  <RocketLaunchIcon className="w-6 h-6" />
                  <span>Start Free Trial</span>
                  <ArrowRightIcon className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </button>
              </Link>
              <p className="text-white/70 text-sm mt-6">No credit card required • 14-day free trial</p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative bg-black/50 backdrop-blur-xl border-t border-white/5 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid md:grid-cols-4 gap-12 mb-12">
            <div>
              <div className="flex items-center gap-3 mb-6">
                <span className="text-3xl">🧞‍♂️</span>
                <span className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Computer Genie
                </span>
              </div>
              <p className="text-gray-400">
                Automate anything with visual workflows
              </p>
            </div>
            <div>
              <h3 className="text-white font-bold mb-4">Product</h3>
              <ul className="space-y-3 text-gray-400">
                <li><Link href="/pricing" className="hover:text-white transition-colors">Pricing</Link></li>
                <li><Link href="/" className="hover:text-white transition-colors">Features</Link></li>
                <li><Link href="/" className="hover:text-white transition-colors">Integrations</Link></li>
              </ul>
            </div>
            <div>
              <h3 className="text-white font-bold mb-4">Company</h3>
              <ul className="space-y-3 text-gray-400">
                <li><Link href="/" className="hover:text-white transition-colors">About</Link></li>
                <li><Link href="/" className="hover:text-white transition-colors">Blog</Link></li>
                <li><Link href="/" className="hover:text-white transition-colors">Careers</Link></li>
              </ul>
            </div>
            <div>
              <h3 className="text-white font-bold mb-4">Legal</h3>
              <ul className="space-y-3 text-gray-400">
                <li><Link href="/" className="hover:text-white transition-colors">Privacy</Link></li>
                <li><Link href="/" className="hover:text-white transition-colors">Terms</Link></li>
                <li><Link href="/" className="hover:text-white transition-colors">Security</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-white/5 pt-8 text-center text-gray-400">
            <p>&copy; 2025 Computer Genie. All rights reserved.</p>
          </div>
        </div>
      </footer>

      <style jsx global>{`
        @keyframes gradient {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }
        .animate-gradient {
          animation: gradient 3s ease infinite;
        }
      `}</style>
    </div>
  );
}
