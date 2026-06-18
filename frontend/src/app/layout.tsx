import type { Metadata } from "next";
import "@fortawesome/fontawesome-free/css/all.min.css";
import "mdb-react-ui-kit/dist/css/mdb.min.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Console",
  description: "Multi-LLM RAG System",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-light">{children}</body>
    </html>
  );
}