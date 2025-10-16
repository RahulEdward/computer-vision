import { withAuth } from 'next-auth/middleware';
import { NextResponse } from 'next/server';

export default withAuth(
  function middleware(req) {
    return NextResponse.next();
  },
  {
    callbacks: {
      authorized: ({ token }) => !!token,
    },
  }
);

// Protect these routes - require authentication
// Landing, Pricing, Auth pages are PUBLIC (not in matcher)
export const config = {
  matcher: [
    /*
     * Protect these routes (require authentication):
     * - /dashboard
     * - /settings
     * - /admin
     * - /onboarding
     * - /workflows
     * 
     * Public routes (no auth required):
     * - / (landing page)
     * - /landing
     * - /pricing
     * - /auth/*
     */
    '/dashboard/:path*',
    '/settings/:path*',
    '/admin/:path*',
    '/onboarding/:path*',
    '/workflows/:path*',
  ],
};
