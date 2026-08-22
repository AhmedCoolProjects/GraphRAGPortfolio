'use client';

import { useApiWarmup } from '@/hooks/use-api-warmup';

export function ApiWarmupProvider() {
  useApiWarmup();
  return null;
}
