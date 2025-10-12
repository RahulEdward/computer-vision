import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { registerCoreServices } from '@/services';

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: 'swap',
});

export const metadata: Metadata = {
  title: "Computer Genie Dashboard",
  description: "Next-generation automation dashboard with AI-powered insights, real-time collaboration, and advanced workflow management",
  keywords: ["automation", "workflow", "collaboration", "AI", "dashboard", "productivity"],
  authors: [{ name: "Computer Genie Team" }],
  creator: "Computer Genie",
  publisher: "Computer Genie",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "Computer Genie",
  },
  openGraph: {
    type: "website",
    siteName: "Computer Genie Dashboard",
    title: "Computer Genie Dashboard",
    description: "Next-generation automation dashboard with AI-powered insights",
  },
  twitter: {
    card: "summary_large_image",
    title: "Computer Genie Dashboard",
    description: "Next-generation automation dashboard with AI-powered insights",
  },
};

export const viewport: Viewport = {
  themeColor: "#7c3aed",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

registerCoreServices();

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="apple-touch-icon" href="/icon-192x192.png" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="msapplication-TileColor" content="#7c3aed" />
        <meta name="msapplication-config" content="/browserconfig.xml" />
      </head>
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
