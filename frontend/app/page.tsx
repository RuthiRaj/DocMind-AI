'use client';

import React from 'react';
import Link from 'next/link';
import { Database, ArrowRight, Brain, Zap, Shield, Search, MessageSquare, Terminal } from 'lucide-react';
import ThemeToggle from '@/components/common/ThemeToggle';

export default function Home() {
  const features = [
    {
      icon: Zap,
      title: 'Smart PDF Parser',
      desc: 'Extracts formatted text blocks and structural document pages cleanly.',
    },
    {
      icon: Brain,
      title: 'Semantic Text Chunking',
      desc: 'Subdivides documents into clean context window boundaries.',
    },
    {
      icon: Search,
      title: 'FAISS Search Engine',
      desc: 'Retrieves relevant sections using cosine similarity score matchings.',
    },
    {
      icon: MessageSquare,
      title: 'Grounded Chat RAG',
      desc: 'Answers user questions strictly with citation source references.',
    },
  ];

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 dark:bg-[#090d16] transition-colors duration-200">
      {/* Header navbar */}
      <header className="flex items-center justify-between h-16 px-6 md:px-12 border-b border-border bg-white/70 dark:bg-slate-900/70 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <Database className="w-6 h-6 text-primary" />
          <span className="font-bold text-xl tracking-tight bg-gradient-to-r from-primary to-violet-500 bg-clip-text text-transparent">
            DocMind AI
          </span>
        </div>
        <div className="flex items-center gap-4">
          <ThemeToggle />
          <Link
            href="/dashboard"
            className="flex items-center gap-1.5 px-4.5 py-2 bg-primary hover:bg-primary/95 text-primary-foreground font-semibold text-sm rounded-lg shadow-sm transition-colors cursor-pointer"
          >
            <span>Enter App</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </header>

      {/* Main hero */}
      <main className="flex-1 flex flex-col items-center">
        {/* Hero Section */}
        <section className="w-full max-w-5xl px-6 pt-20 pb-16 text-center space-y-6 flex flex-col items-center">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-primary/10 text-primary text-xs font-semibold rounded-full border border-primary/25">
            <Terminal className="w-3.5 h-3.5" />
            <span>FastAPI + Next.js App Router</span>
          </div>

          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-foreground max-w-3xl leading-tight">
            Production-Grade{' '}
            <span className="bg-gradient-to-r from-primary via-violet-500 to-indigo-500 bg-clip-text text-transparent">
              Semantic Document RAG
            </span>{' '}
            Engine
          </h1>
          
          <p className="text-lg text-muted-foreground max-w-2xl leading-relaxed">
            Ingest large PDF files, chunk text paragraphs semantically, build vector index databases, and perform grounded AI RAG completions with full precise page citation sources.
          </p>

          <div className="pt-4 flex flex-col sm:flex-row gap-4 justify-center w-full max-w-md">
            <Link
              href="/dashboard"
              className="flex items-center justify-center gap-2 px-8 py-3 bg-primary hover:bg-primary/95 text-primary-foreground font-semibold rounded-xl shadow-lg hover:shadow-primary/20 transition-all cursor-pointer"
            >
              <span>Get Started</span>
              <ArrowRight className="w-5 h-5" />
            </Link>
            <Link
              href="/health"
              className="flex items-center justify-center gap-2 px-8 py-3 bg-transparent hover:bg-muted border border-border text-foreground font-semibold rounded-xl transition-all cursor-pointer"
            >
              <span>Verify Health</span>
            </Link>
          </div>
        </section>

        {/* Feature Grid */}
        <section className="w-full max-w-5xl px-6 py-16 border-t border-border/80">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((f, i) => {
              const Icon = f.icon;
              return (
                <div
                  key={i}
                  className="p-6 bg-white dark:bg-slate-900 border border-border rounded-2xl flex flex-col space-y-3 hover:border-primary/45 transition-all shadow-sm hover:shadow-md"
                >
                  <div className="p-3 bg-primary/10 rounded-xl w-fit text-primary">
                    <Icon className="w-6 h-6" />
                  </div>
                  <h3 className="font-semibold text-lg text-foreground">{f.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
                </div>
              );
            })}
          </div>
        </section>

        {/* Pipeline demonstration flow visualizer */}
        <section className="w-full max-w-5xl px-6 py-16 border-t border-border/80 flex flex-col items-center space-y-8">
          <div className="text-center space-y-2">
            <h2 className="text-3xl font-bold tracking-tight text-foreground">Pipeline Flow</h2>
            <p className="text-muted-foreground">Trace how documents are ingested and compiled step-by-step</p>
          </div>

          <div className="flex flex-col md:flex-row items-center gap-4 w-full justify-between max-w-4xl pt-4">
            {[
              { label: '1. Ingestion', desc: 'PDF File upload' },
              { label: '2. Text Parsing', desc: 'Structure pages' },
              { label: '3. Semantic Chunking', desc: 'Segment paragraphs' },
              { label: '4. Vector indexing', desc: 'Embed & FAISS Store' },
              { label: '5. RAG completion', desc: 'Grounded AI Chat' },
            ].map((step, i, arr) => (
              <React.Fragment key={i}>
                <div className="flex flex-col items-center text-center p-4 bg-white dark:bg-slate-900 border border-border rounded-xl w-44 shadow-sm hover:scale-[1.02] transition-transform">
                  <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary text-primary-foreground font-bold text-sm">
                    {i + 1}
                  </div>
                  <h4 className="font-semibold text-sm mt-3 text-foreground">{step.label}</h4>
                  <p className="text-xs text-muted-foreground mt-1">{step.desc}</p>
                </div>
                {i < arr.length - 1 && (
                  <div className="hidden md:block text-muted-foreground text-2xl font-light">
                    →
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="py-8 px-6 text-center border-t border-border bg-white dark:bg-slate-900/50">
        <p className="text-xs text-muted-foreground">
          © {new Date().getFullYear()} DocMind AI. Built following clean architecture design systems. All rights reserved.
        </p>
      </footer>
    </div>
  );
}
