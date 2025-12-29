/**
 * Logging utilities for Inception ONNX service
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

const currentLevel = LOG_LEVELS[
  (process.env.LOG_LEVEL?.toLowerCase() as LogLevel) || 'info'
] ?? LOG_LEVELS.info;

function formatTimestamp(): string {
  return new Date().toISOString();
}

function shouldLog(level: LogLevel): boolean {
  return LOG_LEVELS[level] >= currentLevel;
}

function formatMessage(level: LogLevel, message: string, meta?: Record<string, unknown>): string {
  const timestamp = formatTimestamp();
  const levelUpper = level.toUpperCase().padEnd(5);
  let output = `[${timestamp}] ${levelUpper} ${message}`;

  if (meta && Object.keys(meta).length > 0) {
    output += ` ${JSON.stringify(meta)}`;
  }

  return output;
}

export const logger = {
  debug(message: string, meta?: Record<string, unknown>): void {
    if (shouldLog('debug')) {
      console.debug(formatMessage('debug', message, meta));
    }
  },

  info(message: string, meta?: Record<string, unknown>): void {
    if (shouldLog('info')) {
      console.info(formatMessage('info', message, meta));
    }
  },

  warn(message: string, meta?: Record<string, unknown>): void {
    if (shouldLog('warn')) {
      console.warn(formatMessage('warn', message, meta));
    }
  },

  error(message: string, error?: Error | unknown, meta?: Record<string, unknown>): void {
    if (shouldLog('error')) {
      const errorMeta = { ...meta };
      if (error instanceof Error) {
        errorMeta.error = error.message;
        errorMeta.stack = error.stack;
      } else if (error) {
        errorMeta.error = String(error);
      }
      console.error(formatMessage('error', message, errorMeta));
    }
  },
};

/**
 * Timer utility for measuring operations
 */
export class Timer {
  private startTime: number;
  private checkpoints: Map<string, number> = new Map();

  constructor(private name: string) {
    this.startTime = performance.now();
  }

  checkpoint(label: string): number {
    const now = performance.now();
    const elapsed = now - this.startTime;
    this.checkpoints.set(label, elapsed);
    return elapsed;
  }

  elapsed(): number {
    return performance.now() - this.startTime;
  }

  log(message?: string): void {
    const elapsed = this.elapsed();
    logger.info(`[Timer:${this.name}] ${message || 'completed'}`, {
      elapsed_ms: elapsed.toFixed(2),
    });
  }

  getCheckpoints(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const [label, time] of this.checkpoints) {
      result[label] = parseFloat(time.toFixed(2));
    }
    return result;
  }
}
