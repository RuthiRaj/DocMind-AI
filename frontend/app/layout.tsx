import type { Metadata } from 'next';
import './globals.css';

import { ThemeProvider } from '@/providers/ThemeProvider';
import { QueryClientProvider } from '@/providers/QueryClientProvider';
import { ToastProvider } from '@/providers/ToastProvider';

export const metadata: Metadata = {
  title: 'DocMind AI - Production-Grade RAG Engine',
  description: 'Grounded AI Chat, Text Chunking, Vector Embeddings, and FAISS Indexing for PDF documents.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className="h-full antialiased"
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col">
        <QueryClientProvider>
          <ThemeProvider>
            <ToastProvider>
              {children}
            </ToastProvider>
          </ThemeProvider>
        </QueryClientProvider>
      </body>
    </html>
  );
}
