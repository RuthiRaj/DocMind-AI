import axios, { AxiosError } from 'axios';
import { API_BASE_URL, API_TIMEOUT } from '@/constants/api';
import { ApiError } from '@/types/Api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

export function handleApiError(error: unknown): ApiError {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{ detail?: string | unknown }>;
    
    // Map offline/timeout errors to helpful instructions
    if (axiosError.code === 'ERR_NETWORK') {
      return {
        message: 'Could not connect to the backend server. Please verify the FastAPI service is active on port 8000.',
        statusCode: 503,
      };
    }
    if (axiosError.code === 'ECONNABORTED') {
      return {
        message: 'The request timed out. The backend server took too long to complete this action.',
        statusCode: 408,
      };
    }

    const responseStatus = axiosError.response?.status;
    const responseData = axiosError.response?.data as
      | { detail?: string | unknown; message?: string | unknown }
      | undefined;
    const backendMessage = responseData?.detail ?? responseData?.message;
    const requestUrl = axiosError.config?.url ?? '';

    let friendlyMessage = 'An unexpected server error occurred.';
    if (responseStatus === 413) {
      if (requestUrl.includes('/chat/')) {
        friendlyMessage =
          (typeof backendMessage === 'string' && backendMessage) ||
          'Document context too large for the AI model — try a more specific question or a shorter document.';
      } else {
        friendlyMessage =
          (typeof backendMessage === 'string' && backendMessage) ||
          'The uploaded PDF file is too large (maximum allowed size is 20MB).';
      }
    } else if (backendMessage) {
      friendlyMessage = typeof backendMessage === 'string' ? backendMessage : JSON.stringify(backendMessage);
    } else if (responseStatus === 404) {
      friendlyMessage = 'The requested resource or document was not found.';
    } else if (responseStatus === 500) {
      friendlyMessage = 'Internal Server Error. The backend experienced an unexpected exception during execution.';
    } else {
      friendlyMessage = axiosError.message || friendlyMessage;
    }

    return {
      message: friendlyMessage,
      statusCode: responseStatus,
      details: axiosError.response?.data,
    };
  }
  
  return {
    message: error instanceof Error ? error.message : 'An unknown network error occurred.',
  };
}

export default apiClient;
