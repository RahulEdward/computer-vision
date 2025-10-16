import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  /* config options here */
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
    turbo: {
      root: process.cwd(),
    },
  },
};

export default nextConfig;
