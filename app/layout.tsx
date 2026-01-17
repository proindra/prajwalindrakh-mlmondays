import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  metadataBase: new URL("https://neuralhive.github.io/ml-mondays"),
  title: "Neural Insights | Neural Hive - Machine Learning & AI Insights",
  description: "Weekly machine learning insights, AI research, and neural network tutorials from the Neural Hive community. Explore cutting-edge ML content every Monday.",
  keywords: "machine learning, artificial intelligence, neural networks, deep learning, AI research, ML tutorials, data science",
  authors: [{ name: "Neural Hive" }],
  creator: "Neural Hive",
  publisher: "Neural Hive",
  openGraph: {
    title: "Neural Insights | Neural Hive",
    description: "Weekly machine learning insights and AI research from the Neural Hive community",
    url: "https://neuralhive.github.io/ml-mondays",
    siteName: "Neural Insights",
    images: [{
      url: "/neural-hive-logo.svg",
      width: 1200,
      height: 630,
      alt: "ML Mondays - Neural Hive"
    }],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Neural Insights | Neural Hive",
    description: "Weekly machine learning insights and AI research",
    images: ["/neural-hive-logo.svg"],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="canonical" href="https://neuralhive.github.io/ml-mondays" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body className={`${inter.className}`}>
        <Header />
        <main className="main-layout">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}