/**
 * Hardware detection and execution provider selection
 * Prioritizes CUDA on DGX Spark, falls back to CPU
 */

import { execSync } from 'child_process';
import type { ExecutionProvider, HardwareInfo } from '../types';
import { logger } from './logger';
import { settings } from './config';

let cachedHardwareInfo: HardwareInfo | null = null;

/**
 * Check if CUDA is available via nvidia-smi
 */
function checkNvidiaSmi(): { available: boolean; deviceName?: string; memoryTotal?: number; cudaVersion?: string } {
  try {
    const output = execSync('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits', {
      encoding: 'utf-8',
      timeout: 5000,
    }).trim();

    if (!output) {
      return { available: false };
    }

    const [deviceName, memoryStr, driverVersion] = output.split(',').map(s => s.trim());
    const memoryTotal = parseInt(memoryStr, 10);

    // Get CUDA version separately
    let cudaVersion: string | undefined;
    try {
      const cudaOutput = execSync('nvidia-smi --query-gpu=driver_version --format=csv,noheader', {
        encoding: 'utf-8',
        timeout: 5000,
      }).trim();
      // Try to get actual CUDA version from nvcc
      try {
        const nvccOutput = execSync('nvcc --version 2>/dev/null | grep "release" | sed "s/.*release \\([0-9.]*\\).*/\\1/"', {
          encoding: 'utf-8',
          timeout: 5000,
        }).trim();
        if (nvccOutput) {
          cudaVersion = nvccOutput;
        }
      } catch {
        // nvcc not available, use driver version as fallback
        cudaVersion = `Driver ${driverVersion}`;
      }
    } catch {
      // Ignore
    }

    return {
      available: true,
      deviceName,
      memoryTotal,
      cudaVersion,
    };
  } catch (error) {
    logger.debug('nvidia-smi not available or failed', { error: String(error) });
    return { available: false };
  }
}

/**
 * Check current GPU memory usage
 */
function getGpuMemoryInfo(): { total?: number; free?: number; used?: number } {
  try {
    const output = execSync('nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits', {
      encoding: 'utf-8',
      timeout: 5000,
    }).trim();

    const [totalStr, freeStr, usedStr] = output.split(',').map(s => s.trim());

    return {
      total: parseInt(totalStr, 10),
      free: parseInt(freeStr, 10),
      used: parseInt(usedStr, 10),
    };
  } catch {
    return {};
  }
}

/**
 * Detect available hardware and select best execution provider
 */
export function detectHardware(): HardwareInfo {
  if (cachedHardwareInfo) {
    return cachedHardwareInfo;
  }

  logger.info('Detecting hardware capabilities...');

  // Check if forced to CPU
  if (settings.forceCpu) {
    logger.info('FORCE_CPU is set, using CPU provider');
    cachedHardwareInfo = {
      provider: 'cpu',
      deviceName: 'CPU (forced)',
    };
    return cachedHardwareInfo;
  }

  // Check for CUDA availability
  const nvidia = checkNvidiaSmi();

  if (nvidia.available) {
    const memInfo = getGpuMemoryInfo();

    cachedHardwareInfo = {
      provider: 'cuda',
      deviceName: nvidia.deviceName || 'NVIDIA GPU',
      memoryTotal: memInfo.total,
      memoryFree: memInfo.free,
      cudaVersion: nvidia.cudaVersion,
    };

    logger.info(`CUDA detected: ${cachedHardwareInfo.deviceName}`, {
      memoryTotal: memInfo.total,
      memoryFree: memInfo.free,
      cudaVersion: nvidia.cudaVersion,
    });

    return cachedHardwareInfo;
  }

  // Fall back to CPU
  logger.info('No CUDA device found, using CPU provider');
  cachedHardwareInfo = {
    provider: 'cpu',
    deviceName: getCpuInfo(),
  };

  return cachedHardwareInfo;
}

/**
 * Get CPU info for logging
 */
function getCpuInfo(): string {
  try {
    // Try to get CPU model on Linux
    const output = execSync('cat /proc/cpuinfo 2>/dev/null | grep "model name" | head -1 | cut -d ":" -f2', {
      encoding: 'utf-8',
      timeout: 2000,
    }).trim();

    if (output) {
      return output;
    }
  } catch {
    // Ignore
  }

  // Fall back to arch info
  try {
    const arch = execSync('uname -m', { encoding: 'utf-8', timeout: 1000 }).trim();
    return `CPU (${arch})`;
  } catch {
    return 'CPU';
  }
}

/**
 * Get ONNX Runtime execution providers in order of preference
 */
export function getExecutionProviders(): string[] {
  const hw = detectHardware();

  if (hw.provider === 'cuda') {
    // CUDA first, with CPU fallback
    return ['CUDAExecutionProvider', 'CPUExecutionProvider'];
  }

  return ['CPUExecutionProvider'];
}

/**
 * Get current execution provider name for display
 */
export function getCurrentProvider(): ExecutionProvider {
  return detectHardware().provider;
}

/**
 * Get detailed hardware info for health/status endpoints
 */
export function getHardwareStatus(): HardwareInfo & { memoryUsed?: number } {
  const hw = detectHardware();

  // Refresh memory info if CUDA
  if (hw.provider === 'cuda') {
    const memInfo = getGpuMemoryInfo();
    return {
      ...hw,
      memoryTotal: memInfo.total,
      memoryFree: memInfo.free,
      memoryUsed: memInfo.used,
    };
  }

  return hw;
}

/**
 * Validate that ONNX runtime can use the selected provider
 */
export async function validateOnnxProviders(): Promise<{ valid: boolean; availableProviders: string[]; error?: string }> {
  try {
    // Dynamic import to handle optional CUDA support
    const ort = await import('onnxruntime-node');
    const available = ort.env?.wasm?.numThreads !== undefined ? ['CPUExecutionProvider'] : [];

    // Try to check for CUDA provider
    // Note: This is a simple check - actual validation happens at session creation
    const hwInfo = detectHardware();
    if (hwInfo.provider === 'cuda') {
      // ONNX Runtime with CUDA should have this
      available.unshift('CUDAExecutionProvider');
    }

    return { valid: true, availableProviders: available };
  } catch (error) {
    return {
      valid: false,
      availableProviders: [],
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
