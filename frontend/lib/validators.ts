import { FILE_LIMITS } from '@/constants/limits';

export function validatePdfFile(file: File): { isValid: boolean; error?: string } {
  if (file.type !== 'application/pdf' && !file.name.endsWith('.pdf')) {
    return { isValid: false, error: 'Only PDF documents are supported.' };
  }
  if (file.size > FILE_LIMITS.MAX_SIZE_BYTES) {
    const sizeMb = FILE_LIMITS.MAX_SIZE_BYTES / (1024 * 1024);
    return { isValid: false, error: `File size exceeds the limit of ${sizeMb}MB.` };
  }
  return { isValid: true };
}
