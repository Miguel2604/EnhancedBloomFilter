import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

type ThroughputFormatOptions = {
  millionDecimals?: number
  thousandDecimals?: number
}

export function formatThroughput(value?: number, options?: ThroughputFormatOptions): string {
  const val = Math.max(0, value ?? 0)
  const millionDecimals = options?.millionDecimals ?? 2
  const thousandDecimals = options?.thousandDecimals ?? 1

  if (val >= 1_000_000) {
    return `${(val / 1_000_000).toFixed(millionDecimals)}M`
  }

  if (val >= 1_000) {
    return `${(val / 1_000).toFixed(thousandDecimals)}K`
  }

  if (val === 0) {
    return '0'
  }

  return val < 10 ? val.toFixed(2) : val.toFixed(0)
}
