'use client';

import { useEffect, useRef } from 'react';

const API_URL = process.env.NEXT_PUBLIC_CHAT_API_URL || "http://127.0.0.1:8000";

export function useApiWarmup() {
  const hasWarmedUp = useRef(false);

  useEffect(() => {
    if (hasWarmedUp.current) return;
    hasWarmedUp.current = true;

    const warmupApi = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        await fetch(`${API_URL}/health`, {
          method: 'GET',
          signal: controller.signal,
        });
        
        clearTimeout(timeoutId);
      } catch (error) {
        // Silently fail - warm-up is just a best effort
        console.log('API warm-up attempted');
      }
    };

    warmupApi();
  }, []);
}
