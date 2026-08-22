import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ApiWarmupProvider } from "@/components/providers/api-warmup-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Ahmed Bargady | PhD Student in AI & Cybersecurity",
  description:
    "PhD Student at UM6P specializing in AI-driven Intrusion Detection Systems. Research Engineer with expertise in LLMs, Cybersecurity, and Data Science. Building the future of AI-powered security solutions.",
  keywords: [
    "Ahmed Bargady",
    "AI",
    "Cybersecurity",
    "PhD Student",
    "Intrusion Detection Systems",
    "Machine Learning",
    "LLM",
    "Research Engineer",
    "UM6P",
    "Data Science",
  ],
  authors: [{ name: "Ahmed Bargady" }],
  creator: "Ahmed Bargady",
  publisher: "Ahmed Bargady",
  openGraph: {
    title: "Ahmed Bargady | PhD Student in AI & Cybersecurity",
    description:
      "PhD Student at UM6P specializing in AI-driven Intrusion Detection Systems. Research Engineer with expertise in LLMs, Cybersecurity, and Data Science.",
    url: "https://ahmedbargady.com",
    siteName: "Ahmed Bargady Portfolio",
    locale: "en_US",
    type: "website",
    images: [
      {
        url: "/me.png",
        width: 1200,
        height: 630,
        alt: "Ahmed Bargady - PhD Student in AI & Cybersecurity",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Ahmed Bargady | PhD Student in AI & Cybersecurity",
    description:
      "PhD Student at UM6P specializing in AI-driven Intrusion Detection Systems. Research Engineer with expertise in LLMs, Cybersecurity, and Data Science.",
    images: ["/me.png"],
    creator: "@ahmedbargady",
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
  verification: {
    google: "your-google-verification-code",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ApiWarmupProvider />
        {children}
      </body>
    </html>
  );
}
