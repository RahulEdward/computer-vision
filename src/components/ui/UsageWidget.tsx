'use client';

import { motion } from 'framer-motion';

interface UsageWidgetProps {
  title: string;
  current: number;
  limit: number;
  unit?: string;
}

export default function UsageWidget({ title, current, limit, unit = '' }: UsageWidgetProps) {
  const percentage = limit === -1 ? 0 : (current / limit) * 100;
  const isUnlimited = limit === -1;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6"
    >
      <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
      
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-3xl font-bold text-white">
          {current.toLocaleString()}
        </span>
        <span className="text-gray-400">
          {isUnlimited ? 'Unlimited' : `/ ${limit.toLocaleString()} ${unit}`}
        </span>
      </div>

      {!isUnlimited && (
        <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(percentage, 100)}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            className={`h-full rounded-full ${
              percentage > 90
                ? 'bg-red-500'
                : percentage > 70
                ? 'bg-yellow-500'
                : 'bg-green-500'
            }`}
          />
        </div>
      )}

      {!isUnlimited && percentage > 80 && (
        <p className="text-yellow-400 text-sm mt-2">
          ⚠️ Approaching limit
        </p>
      )}
    </motion.div>
  );
}
