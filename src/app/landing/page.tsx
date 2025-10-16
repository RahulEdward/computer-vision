'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { 
  BoltIcon, 
  CloudIcon, 
  ShieldCheckIcon, 
  ChartBarIcon,
  UsersIcon,
  CogIcon 
} from '@heroicons/react/24/outline';

export default function LandingPage() {
  const features = [
    {
      icon: BoltIcon,
      title: 'Lightning Fast',
      description: 'Execute workflows in milliseconds with our optimized engine'
    },
    {
      icon: CloudIcon,
      title: 'Cloud Native',
      description: 'Built for scale with modern cloud infrastructure'
    },
    {
      icon: ShieldCheckIcon,
      title: 'Enterprise Security',
      description: 'Bank-grade encryption and compliance certifications'
    },
    {
      icon: ChartBarIcon,
      title: 'Advanced Analytics',
      description: 'Real-time insights into your automation performance'
    },
    {
      icon: UsersIcon,
      title: 'Team Collaboration',
      description: 'Work together seamlessly with your team'
    },
    {
      icon: CogIcon,
      title: 'Easy Integration',
      description: 'Connect with 1000+ apps and services'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Navigation */}
      <nav className="bg-black/20 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-2xl">🧞‍♂️</span>
              <span className="text-xl font-bold text-white">Computer Genie</span>
            </div>
            <div className="flex items-center space-x-4">
              <Link href="/pricing" className="text-gray-300 hover:text-white">
                Pricing
              </Link>
              <Link href="/auth/login" className="text-gray-300 hover:text-white">
                Login
              </Link>
              <Link href="/auth/signup">
                <button className="px-6 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700">
                  Get Started
                </button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <h1 className="text-6xl font-bold text-white mb-6">
            Automate Anything,
            <br />
            <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Build Faster
            </span>
          </h1>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            The most powerful workflow automation platform. Visual programming meets AI to help you build complex automations without code.
          </p>
          <div className="flex justify-center space-x-4">
            <Link href="/auth/signup">
              <button className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg text-lg font-medium hover:from-purple-700 hover:to-pink-700">
                Start Free Trial
              </button>
            </Link>
            <Link href="/pricing">
              <button className="px-8 py-4 bg-white/10 border border-white/20 text-white rounded-lg text-lg font-medium hover:bg-white/20">
                View Pricing
              </button>
            </Link>
          </div>
        </motion.div>

        {/* Demo Video/Image Placeholder */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mt-16 bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-8 aspect-video flex items-center justify-center"
        >
          <p className="text-gray-400 text-xl">🎬 Product Demo Video</p>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="max-w-7xl mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-white mb-4">
            Everything You Need
          </h2>
          <p className="text-xl text-gray-300">
            Powerful features to supercharge your automation
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6 hover:border-purple-500/50 transition-all"
            >
              <feature.icon className="h-12 w-12 text-purple-400 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-400">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Social Proof */}
      <section className="max-w-7xl mx-auto px-6 py-20">
        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-12 text-center">
          <h2 className="text-3xl font-bold text-white mb-8">
            Trusted by Teams Worldwide
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <div className="text-4xl font-bold text-purple-400 mb-2">10K+</div>
              <div className="text-gray-400">Active Users</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-purple-400 mb-2">1M+</div>
              <div className="text-gray-400">Workflows Executed</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-purple-400 mb-2">99.9%</div>
              <div className="text-gray-400">Uptime</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="max-w-7xl mx-auto px-6 py-20">
        <div className="bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl p-12 text-center">
          <h2 className="text-4xl font-bold text-white mb-4">
            Ready to Get Started?
          </h2>
          <p className="text-xl text-white/90 mb-8">
            Join thousands of teams automating their workflows
          </p>
          <Link href="/auth/signup">
            <button className="px-8 py-4 bg-white text-purple-600 rounded-lg text-lg font-medium hover:bg-gray-100">
              Start Free Trial - No Credit Card Required
            </button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-black/40 backdrop-blur-xl border-t border-white/10 py-12">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <span className="text-2xl">🧞‍♂️</span>
                <span className="text-lg font-bold text-white">Computer Genie</span>
              </div>
              <p className="text-gray-400">
                Automate anything with visual workflows
              </p>
            </div>
            <div>
              <h3 className="text-white font-semibold mb-4">Product</h3>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/pricing">Pricing</Link></li>
                <li><Link href="/">Features</Link></li>
                <li><Link href="/">Integrations</Link></li>
              </ul>
            </div>
            <div>
              <h3 className="text-white font-semibold mb-4">Company</h3>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/">About</Link></li>
                <li><Link href="/">Blog</Link></li>
                <li><Link href="/">Careers</Link></li>
              </ul>
            </div>
            <div>
              <h3 className="text-white font-semibold mb-4">Legal</h3>
              <ul className="space-y-2 text-gray-400">
                <li><Link href="/">Privacy</Link></li>
                <li><Link href="/">Terms</Link></li>
                <li><Link href="/">Security</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-white/10 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2025 Computer Genie. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
