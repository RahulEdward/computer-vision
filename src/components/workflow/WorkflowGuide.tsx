'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  QuestionMarkCircleIcon,
  XMarkIcon,
  CursorArrowRaysIcon,
  ArrowsPointingOutIcon,
  PencilSquareIcon,
  TrashIcon,
  LinkIcon,
} from '@heroicons/react/24/outline';

export default function WorkflowGuide() {
  const [isOpen, setIsOpen] = useState(false);

  const guides = [
    {
      icon: CursorArrowRaysIcon,
      title: 'Add Node',
      description: 'Left sidebar se koi bhi node select karo aur click karo. Node canvas pe add ho jayega.',
      color: 'from-blue-500 to-blue-600',
    },
    {
      icon: LinkIcon,
      title: 'Connect Nodes',
      description: 'Node pe hover karo → Right side ka dot pakdo → Dusre node tak drag karo → Connection ban jayega!',
      color: 'from-green-500 to-green-600',
    },
    {
      icon: PencilSquareIcon,
      title: 'Edit Node',
      description: 'Node pe click karo → Right sidebar mein settings khulegi → Properties change karo → Save karo.',
      color: 'from-purple-500 to-purple-600',
    },
    {
      icon: ArrowsPointingOutIcon,
      title: 'Move Node',
      description: 'Node ko drag karo aur jahan chahiye wahan drop karo. Canvas ko pan karne ke liye empty space drag karo.',
      color: 'from-orange-500 to-orange-600',
    },
    {
      icon: TrashIcon,
      title: 'Delete Node',
      description: 'Node select karo (click karke) aur Delete ya Backspace key press karo. Connection bhi delete ho jayega.',
      color: 'from-red-500 to-red-600',
    },
  ];

  return (
    <>
      {/* Help Button - Top Center */}
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setIsOpen(!isOpen)}
        className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg shadow-lg hover:shadow-xl transition-all"
      >
        <QuestionMarkCircleIcon className="h-5 w-5" />
        <span className="font-medium">Help Guide</span>
      </motion.button>

      {/* Guide Panel */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50"
            />

            {/* Panel */}
            <motion.div
              initial={{ opacity: 0, y: -20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20, scale: 0.95 }}
              className="fixed top-20 left-1/2 transform -translate-x-1/2 w-full max-w-4xl bg-white rounded-2xl shadow-2xl z-50 overflow-hidden"
            >
              {/* Header */}
              <div className="bg-gradient-to-r from-purple-600 to-pink-600 px-6 py-4 flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-white">Workflow Builder Guide</h2>
                  <p className="text-white/90 text-sm">Yeh sab kuch kar sakte hain aap!</p>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-2 hover:bg-white/20 rounded-lg transition-colors"
                >
                  <XMarkIcon className="h-6 w-6 text-white" />
                </button>
              </div>

              {/* Content */}
              <div className="p-6 max-h-[70vh] overflow-y-auto">
                {/* Quick Tips */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">⚡ Quick Tips</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                      <div className="text-2xl">🖱️</div>
                      <div>
                        <p className="font-medium text-gray-900">Mouse Wheel</p>
                        <p className="text-sm text-gray-600">Zoom in/out karne ke liye</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg">
                      <div className="text-2xl">🎯</div>
                      <div>
                        <p className="font-medium text-gray-900">Mini Map</p>
                        <p className="text-sm text-gray-600">Bottom-right corner mein overview</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3 p-3 bg-purple-50 rounded-lg">
                      <div className="text-2xl">⌨️</div>
                      <div>
                        <p className="font-medium text-gray-900">Keyboard Shortcuts</p>
                        <p className="text-sm text-gray-600">Delete = Node remove, Ctrl+Z = Undo</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3 p-3 bg-orange-50 rounded-lg">
                      <div className="text-2xl">💾</div>
                      <div>
                        <p className="font-medium text-gray-900">Auto Save</p>
                        <p className="text-sm text-gray-600">Changes automatically save hote hain</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Step by Step Guide */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">📚 Step-by-Step Guide</h3>
                  <div className="space-y-4">
                    {guides.map((guide, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="flex items-start space-x-4 p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors"
                      >
                        <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${guide.color} flex items-center justify-center flex-shrink-0`}>
                          <guide.icon className="h-6 w-6 text-white" />
                        </div>
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-900 mb-1">{guide.title}</h4>
                          <p className="text-sm text-gray-600">{guide.description}</p>
                        </div>
                        <div className="text-2xl font-bold text-gray-300">{index + 1}</div>
                      </motion.div>
                    ))}
                  </div>
                </div>

                {/* Available Nodes */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">🎨 Available Node Types</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                      <div className="font-medium text-green-900">🎯 Triggers</div>
                      <p className="text-xs text-green-700">Manual, Webhook, Schedule</p>
                    </div>
                    <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                      <div className="font-medium text-blue-900">⚡ Actions</div>
                      <p className="text-xs text-blue-700">HTTP, Transform, Condition</p>
                    </div>
                    <div className="p-3 bg-purple-50 border border-purple-200 rounded-lg">
                      <div className="font-medium text-purple-900">🔄 Transform</div>
                      <p className="text-xs text-purple-700">Data manipulation</p>
                    </div>
                    <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                      <div className="font-medium text-gray-900">🔧 Core</div>
                      <p className="text-xs text-gray-700">Basic functionality</p>
                    </div>
                    <div className="p-3 bg-indigo-50 border border-indigo-200 rounded-lg">
                      <div className="font-medium text-indigo-900">🔌 Integration</div>
                      <p className="text-xs text-indigo-700">External services</p>
                    </div>
                    <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <div className="font-medium text-yellow-900">🛠️ Utility</div>
                      <p className="text-xs text-yellow-700">Helper functions</p>
                    </div>
                  </div>
                </div>

                {/* Example Workflow */}
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">💡 Example Workflow</h3>
                  <div className="space-y-2 text-sm text-gray-700">
                    <p className="flex items-center space-x-2">
                      <span className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-xs">1</span>
                      <span><strong>Manual Trigger</strong> - Workflow start karo</span>
                    </p>
                    <p className="flex items-center space-x-2 ml-3">
                      <span className="text-gray-400">↓</span>
                    </p>
                    <p className="flex items-center space-x-2">
                      <span className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs">2</span>
                      <span><strong>HTTP Request</strong> - API se data fetch karo</span>
                    </p>
                    <p className="flex items-center space-x-2 ml-3">
                      <span className="text-gray-400">↓</span>
                    </p>
                    <p className="flex items-center space-x-2">
                      <span className="w-6 h-6 bg-orange-500 rounded-full flex items-center justify-center text-white text-xs">3</span>
                      <span><strong>Condition</strong> - Data check karo</span>
                    </p>
                    <p className="flex items-center space-x-2 ml-3">
                      <span className="text-gray-400">↓</span>
                    </p>
                    <p className="flex items-center space-x-2">
                      <span className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center text-white text-xs">4</span>
                      <span><strong>Transform</strong> - Data process karo</span>
                    </p>
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="bg-gray-50 px-6 py-4 flex items-center justify-between border-t border-gray-200">
                <p className="text-sm text-gray-600">
                  Need more help? Check the documentation
                </p>
                <button
                  onClick={() => setIsOpen(false)}
                  className="px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:shadow-lg transition-all"
                >
                  Got it!
                </button>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
