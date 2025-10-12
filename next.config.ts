import type { NextConfig } from "next";
// @ts-ignore
import withPWA from 'next-pwa';

const nextConfig: NextConfig = {
  experimental: {
    optimizePackageImports: ['@heroicons/react', 'framer-motion', '@react-three/fiber', '@react-three/drei'],
    turbo: {
      rules: {
        '*.svg': {
          loaders: ['@svgr/webpack'],
          as: '*.js',
        },
      },
    },
  },
  images: {
    domains: ['localhost'],
    formats: ['image/webp', 'image/avif'],
  },
  webpack: (config, { isServer }) => {
    // Handle browser-only modules
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
        canvas: false,
      };
    }

    // Handle WebRTC and other browser-only modules for SSR
    if (isServer) {
      config.externals = [...(config.externals || []), 'simple-peer', 'y-webrtc', 'yjs'];
    }

    // Optimize bundle splitting for better performance
    config.optimization = {
      ...config.optimization,
      splitChunks: {
        chunks: 'all',
        cacheGroups: {
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            chunks: 'all',
          },
          three: {
            test: /[\\/]node_modules[\\/](three|@react-three)[\\/]/,
            name: 'three',
            chunks: 'all',
          },
          monaco: {
            test: /[\\/]node_modules[\\/](@monaco-editor|monaco-editor)[\\/]/,
            name: 'monaco',
            chunks: 'all',
          },
          tensorflow: {
            test: /[\\/]node_modules[\\/]@tensorflow[\\/]/,
            name: 'tensorflow',
            chunks: 'all',
          },
        },
      },
    };

    return config;
  },
  compress: true,
  poweredByHeader: false,
  generateEtags: false,
  httpAgentOptions: {
    keepAlive: true,
  },
};

const pwaConfig = withPWA({
  dest: 'public',
  register: true,
  skipWaiting: true,
  disable: process.env.NODE_ENV === 'development',
  runtimeCaching: [
    {
      urlPattern: /^https?.*/,
      handler: 'NetworkFirst',
      options: {
        cacheName: 'offlineCache',
        expiration: {
          maxEntries: 200,
          maxAgeSeconds: 24 * 60 * 60, // 24 hours
        },
        networkTimeoutSeconds: 10,
      },
    },
    {
      urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|avif)$/,
      handler: 'CacheFirst',
      options: {
        cacheName: 'images',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
        },
      },
    },
    {
      urlPattern: /\.(?:js|css|woff|woff2|ttf|eot)$/,
      handler: 'StaleWhileRevalidate',
      options: {
        cacheName: 'static-resources',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 7 * 24 * 60 * 60, // 7 days
        },
      },
    },
  ],
  buildExcludes: [/middleware-manifest\.json$/],
  maximumFileSizeToCacheInBytes: 5000000, // 5MB
});

export default pwaConfig(nextConfig);
