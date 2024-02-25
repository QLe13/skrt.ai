import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Headers from "./components/Header";
import Provider from "./Provider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "sKrt.ai",
  description: "sKrt.ai",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Provider>
        <Headers />
        {children}
        </Provider>
        </body>
    </html>
  );
}
