import type { Metadata } from "next";
import { Geist, Geist_Mono } from 'next/font/google';
import { Sidebar } from "@/components/sidebar";
import { SimulationProvider } from "@/context/simulation-context";
import { StreamProvider } from "@/context/stream-context";
import { ComparisonProvider } from "@/context/comparison-context";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Bloom Filter Simulation",
  description: "Interactive dashboard for Bloom Filter performance analysis",
    generator: 'v0.app'
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-white`}
      >
        <SimulationProvider>
          <StreamProvider>
            <ComparisonProvider>
              <div className="flex min-h-screen">
                <Sidebar />
                <div className="flex-1 pl-20">
                  {children}
                </div>
              </div>
            </ComparisonProvider>
          </StreamProvider>
        </SimulationProvider>
      </body>
    </html>
  );
}
