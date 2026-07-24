export const PIPELINE_STAGES = [
  { key: 'upload', label: 'Upload Ingestion', desc: 'Saves the raw PDF document into storage directory paths.' },
  { key: 'processing', label: 'PDF Processing', desc: 'Parses pages structure layouts and extracts raw clean text.' },
  { key: 'chunking', label: 'Smart Chunking', desc: 'Subdivides text content into semantic sentences paragraphs.' },
  { key: 'embedding', label: 'Embeddings Generation', desc: 'Encodes text segments into high-dimensional vector representations.' },
  { key: 'indexing', label: 'Index Compilation', desc: 'Stores vectors inside a FAISS retrieval search database.' },
];

export const STAGE_STYLES = {
  waiting: {
    bg: 'bg-slate-100 dark:bg-slate-800',
    text: 'text-slate-500 dark:text-slate-400',
    border: 'border-slate-200 dark:border-slate-700',
    badge: 'secondary',
    label: 'Waiting',
  },
  running: {
    bg: 'bg-blue-50 dark:bg-blue-950/30',
    text: 'text-blue-600 dark:text-blue-400',
    border: 'border-blue-200 dark:border-blue-800',
    badge: 'info',
    label: 'Processing',
  },
  completed: {
    bg: 'bg-emerald-50 dark:bg-emerald-950/30',
    text: 'text-emerald-600 dark:text-emerald-400',
    border: 'border-emerald-200 dark:border-emerald-800',
    badge: 'success',
    label: 'Completed',
  },
  failed: {
    bg: 'bg-red-50 dark:bg-red-950/30',
    text: 'text-red-600 dark:text-red-400',
    border: 'border-red-200 dark:border-red-800',
    badge: 'destructive',
    label: 'Failed',
  },
};
