export const ROUTES = {
  LANDING: '/',
  DASHBOARD: '/dashboard',
  UPLOAD: '/upload',
  DOCUMENTS: '/documents',
  DOCUMENT_DETAILS: (id: string) => `/documents/${id}`,
  DOCUMENT_PIPELINE: (id: string) => `/documents/${id}/pipeline`,
  RETRIEVAL_PLAYGROUND: (id: string) => `/retrieve/${id}`,
  CHAT: (id: string) => `/chat/${id}`,
  HEALTH: '/health',
  MAINTENANCE: '/maintenance',
};

export const NAVIGATION_ITEMS = [
  { label: 'Dashboard', path: ROUTES.DASHBOARD, icon: 'LayoutDashboard' },
  { label: 'Upload PDF', path: ROUTES.UPLOAD, icon: 'UploadCloud' },
  { label: 'Documents', path: ROUTES.DOCUMENTS, icon: 'FileText' },
  { label: 'Health Status', path: ROUTES.HEALTH, icon: 'Activity' },
  { label: 'Maintenance', path: ROUTES.MAINTENANCE, icon: 'Settings' },
];
