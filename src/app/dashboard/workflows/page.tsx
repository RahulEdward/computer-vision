'use client';

import { useEffect } from 'react';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import FullWorkflowBuilder from '@/components/workflow/FullWorkflowBuilder';

export default function WorkflowPage() {
  const { data: session, status } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/auth/login');
    }
  }, [status, router]);

  if (status === 'loading') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-gray-800 text-xl">Loading...</div>
      </div>
    );
  }

  if (!session) {
    return null;
  }

  return <FullWorkflowBuilder />;
}
