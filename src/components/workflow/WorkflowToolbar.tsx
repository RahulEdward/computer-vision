'use client';

import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import {
  ArrowLeftIcon,
  ArrowsPointingInIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';

interface WorkflowToolbarProps {
  onFitView?: () => void;
  onDeleteSelected?: () => void;
  hasSelectedNodes?: boolean;
}

export default function WorkflowToolbar({ 
  onFitView, 
  onDeleteSelected,
  hasSelectedNodes = false 
}: WorkflowToolbarProps) {
  const router = useRouter();

  return (
    <div className="fixed top-4 left-4 z-40 flex items-center gap-2">
      {/* Back to Dashboard */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={() => router.push('/dashboard')}
        className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-gray-700 to-gray-800 hover:from-gray-800 hover:to-gray-900 text-white rounded-lg shadow-md transition-all text-sm font-medium"
      >
        <ArrowLeftIcon className="h-4 w-4" />
        <span>Dashboard</span>
      </motion.button>

      {/* Fit to Screen */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={onFitView}
        className="flex items-center gap-2 px-4 py-2.5 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 rounded-lg shadow-sm transition-all text-sm font-medium"
        title="Fit to screen"
      >
        <ArrowsPointingInIcon className="h-4 w-4" />
        <span>Fit View</span>
      </motion.button>

      {/* Delete Selected */}
      {hasSelectedNodes && (
        <motion.button
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={onDeleteSelected}
          className="flex items-center gap-2 px-4 py-2.5 bg-red-500 hover:bg-red-600 text-white rounded-lg shadow-sm transition-all text-sm font-medium"
        >
          <TrashIcon className="h-4 w-4" />
          <span>Delete</span>
        </motion.button>
      )}
    </div>
  );
}
