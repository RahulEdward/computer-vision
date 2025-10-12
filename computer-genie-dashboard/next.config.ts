import type { NextConfig } from "next";
// @ts-ignore
import withPWA from 'next-pwa';

const nextConfig: NextConfig = {
  /* config options here */
  experimental: {
    optimizePackageImports: ['@heroicons/react', 'framer-motion'],
  },
  images: {
    domains: ['localhost'],
  },
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      };
    }
    return config;
  },
};

const pwaConfig = withPWA({
  dest: 'public',
  register: true,
  skipWaiting: true,
  runtimeCaching: [
    {
      urlPattern: /^https?.*/,
      handler: 'NetworkFirst',
      options: {
        cacheName: 'offlineCache',
        expiration: {
          maxEntries: 200,
        },
      },
    },
  ],
  buildExcludes: [/middleware-manifest.json$/],
});

export default pwaConfig(nextConfig);
