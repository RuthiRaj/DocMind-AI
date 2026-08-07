'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useDocuments } from '@/hooks/useDocuments';
import MainLayout from '@/components/layout/MainLayout';
import { Card, CardContent } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import Input from '@/components/ui/Input';
import Dialog from '@/components/ui/Dialog';
import Skeleton from '@/components/ui/Skeleton';
import { formatBytes } from '@/lib/formatBytes';
import { formatDate } from '@/lib/formatDate';
import {
  FileText,
  Search,
  Trash2,
  MessageSquare,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Info,
  SlidersHorizontal
} from 'lucide-react';

const LIMIT = 10;

export default function DocumentsPage() {
  const [skip, setSkip] = useState(0);
  const [sortBy, setSortBy] = useState('upload_time');
  const [descending, setDescending] = useState(true);
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Dialog State
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [deleteConfirmName, setDeleteConfirmName] = useState<string | null>(null);

  const {
    documents,
    totalCount,
    listLoading,
    deleteDoc,
    isDeleting,
  } = useDocuments({
    skip,
    limit: LIMIT,
    sort_by: sortBy,
    descending,
    status_filter: statusFilter,
  });

  const currentPage = Math.floor(skip / LIMIT) + 1;
  const totalPages = Math.ceil(totalCount / LIMIT);

  const handlePrevPage = () => {
    setSkip((prev) => Math.max(0, prev - LIMIT));
  };

  const handleNextPage = () => {
    setSkip((prev) => Math.min(totalCount - LIMIT, prev + LIMIT));
  };

  const handleSort = (field: string) => {
    if (sortBy === field) {
      setDescending((prev) => !prev);
    } else {
      setSortBy(field);
      setDescending(true);
    }
    setSkip(0);
  };

  const handleDeleteClick = (id: string, name: string) => {
    setDeleteConfirmId(id);
    setDeleteConfirmName(name);
  };

  const confirmDelete = async () => {
    if (deleteConfirmId) {
      await deleteDoc(deleteConfirmId);
      setDeleteConfirmId(null);
      setDeleteConfirmName(null);
    }
  };

  // Local text filter for search query
  const filteredDocuments = documents.filter((doc) =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <MainLayout>
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-foreground font-sans">Document Lifecycle Management</h2>
          <p className="text-muted-foreground text-sm">
            Inspect metadata properties, verify pipeline progress logs, and trigger safe deletes.
          </p>
        </div>
        <Link href="/upload">
          <Button className="cursor-pointer">Ingest Document</Button>
        </Link>
      </div>

      {/* Filter and Search Bar */}
      <Card>
        <CardContent className="p-4 flex flex-col md:flex-row gap-4 items-center justify-between shrink-0">
          {/* Search Box */}
          <div className="relative w-full md:max-w-xs">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by filename..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>

          {/* Status Filters */}
          <div className="flex flex-wrap items-center gap-2 w-full md:w-auto justify-end">
            <span className="text-xs text-muted-foreground font-semibold flex items-center gap-1.5 mr-2">
              <SlidersHorizontal className="w-3.5 h-3.5" />
              Filter Stage:
            </span>
            {[
              { label: 'All', value: null },
              { label: 'Ingested', value: 'upload' },
              { label: 'Processed', value: 'processing' },
              { label: 'Chunked', value: 'chunking' },
              { label: 'Embedded', value: 'embedding' },
              { label: 'Indexed', value: 'indexing' },
            ].map((filter) => (
              <Button
                key={filter.label}
                variant={statusFilter === filter.value ? 'primary' : 'outline'}
                size="sm"
                className="h-8 text-xs cursor-pointer"
                onClick={() => {
                  setStatusFilter(filter.value);
                  setSkip(0);
                }}
              >
                {filter.label}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Documents Table */}
      <Card className="flex flex-col min-h-[400px]">
        <CardContent className="p-0 overflow-x-auto min-w-0 flex-1">
          {listLoading ? (
            <div className="p-6 space-y-4">
              <Skeleton className="w-full h-12" />
              <Skeleton className="w-full h-12" />
              <Skeleton className="w-full h-12" />
            </div>
          ) : filteredDocuments.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground text-sm">
              <FileText className="w-12 h-12 mb-3 opacity-55" />
              <p className="font-semibold text-foreground">No documents found</p>
              <p className="text-xs mt-1">Upload a PDF file or change filters parameters.</p>
            </div>
          ) : (
            <div className="w-full min-w-[800px]">
              <table className="w-full border-collapse text-left text-sm">
                <thead>
                  <tr className="border-b border-border bg-muted/20 text-muted-foreground select-none">
                    <th
                      className="p-4 font-semibold cursor-pointer hover:text-foreground transition-colors"
                      onClick={() => handleSort('filename')}
                    >
                      Filename {sortBy === 'filename' && (descending ? '↓' : '↑')}
                    </th>
                    <th
                      className="p-4 font-semibold cursor-pointer hover:text-foreground transition-colors"
                      onClick={() => handleSort('upload_time')}
                    >
                      Uploaded {sortBy === 'upload_time' && (descending ? '↓' : '↑')}
                    </th>
                    <th className="p-4 font-semibold">Pages</th>
                    <th className="p-4 font-semibold">Chunks</th>
                    <th className="p-4 font-semibold">Size</th>
                    <th className="p-4 font-semibold">Status Stage</th>
                    <th className="p-4 font-semibold text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60">
                  {filteredDocuments.map((doc) => (
                    <tr key={doc.document_id} className="hover:bg-muted/30 transition-colors">
                      <td className="p-4 font-medium text-foreground truncate max-w-[220px]" title={doc.filename}>
                        {doc.filename}
                      </td>
                      <td className="p-4 text-muted-foreground">{formatDate(doc.upload_time)}</td>
                      <td className="p-4">{doc.total_pages ?? 'N/A'}</td>
                      <td className="p-4">{doc.total_chunks ?? 'N/A'}</td>
                      <td className="p-4">{formatBytes(doc.document_size)}</td>
                      <td className="p-4">
                        <Badge
                          variant={
                            doc.current_pipeline_stage === 'indexing'
                              ? 'success'
                              : doc.current_pipeline_stage === 'failed'
                              ? 'destructive'
                              : 'warning'
                          }
                        >
                          {doc.current_pipeline_stage}
                        </Badge>
                      </td>
                      <td className="p-4 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <Link href={`/documents/${doc.document_id}`}>
                            <Button size="sm" variant="outline" className="p-2 cursor-pointer" title="View details info">
                              <Info className="w-4 h-4" />
                            </Button>
                          </Link>
                          
                          {doc.chat_ready ? (
                            <Link href={`/chat/${doc.document_id}`}>
                              <Button size="sm" variant="primary" className="flex items-center gap-1.5 cursor-pointer">
                                <MessageSquare className="w-3.5 h-3.5" />
                                <span>Chat</span>
                              </Button>
                            </Link>
                          ) : (
                            <Link href={`/documents/${doc.document_id}/pipeline`}>
                              <Button size="sm" variant="outline" className="flex items-center gap-1.5 cursor-pointer">
                                <span>Track</span>
                                <ArrowRight className="w-3.5 h-3.5" />
                              </Button>
                            </Link>
                          )}
                          
                          <Button
                            size="sm"
                            variant="ghost"
                            className="p-2 text-rose-500 hover:bg-rose-500/10 hover:text-rose-600 cursor-pointer"
                            onClick={() => handleDeleteClick(doc.document_id, doc.filename)}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
        
        {/* Pagination controls */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between p-4 border-t border-border shrink-0 text-sm">
            <span className="text-muted-foreground">
              Page {currentPage} of {totalPages} ({totalCount} documents)
            </span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handlePrevPage}
                disabled={currentPage === 1}
                className="flex items-center gap-1 cursor-pointer"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Prev</span>
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleNextPage}
                disabled={currentPage === totalPages}
                className="flex items-center gap-1 cursor-pointer"
              >
                <span>Next</span>
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        )}
      </Card>

      {/* Delete Confirmation Dialog */}
      <Dialog
        isOpen={deleteConfirmId !== null}
        onClose={() => {
          setDeleteConfirmId(null);
          setDeleteConfirmName(null);
        }}
        title="Confirm Document Deletion"
      >
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Are you sure you want to permanently delete the document{' '}
            <strong className="text-foreground">{deleteConfirmName}</strong>?
          </p>
          <div className="bg-rose-500/10 text-rose-600 dark:text-rose-400 p-3 rounded-lg border border-rose-500/20 text-xs flex gap-2">
            <Info className="w-4 h-4 shrink-0 mt-0.5" />
            <span>
              This operation is permanent. All original PDF files, chunk data, embedding vectors, and indexes will be permanently removed.
            </span>
          </div>
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setDeleteConfirmId(null);
                setDeleteConfirmName(null);
              }}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              size="sm"
              loading={isDeleting}
              onClick={confirmDelete}
              className="cursor-pointer"
            >
              Delete Permanently
            </Button>
          </div>
        </div>
      </Dialog>
    </MainLayout>
  );
}
