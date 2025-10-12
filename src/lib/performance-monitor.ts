interface PerformanceMetrics {
  interactionLatency: number;
  renderTime: number;
  memoryUsage: number;
  networkLatency: number;
  fps: number;
  bundleSize: number;
}

export class PerformanceMonitor {
  private metrics: PerformanceMetrics = {
    interactionLatency: 0,
    renderTime: 0,
    memoryUsage: 0,
    networkLatency: 0,
    fps: 0,
    bundleSize: 0
  };

  private observers: PerformanceObserver[] = [];
  private frameCount = 0;
  private lastFrameTime = 0;
  private fpsHistory: number[] = [];

  constructor() {
    this.initializeObservers();
    this.startFPSMonitoring();
    this.monitorMemoryUsage();
  }

  private initializeObservers() {
    // Monitor Long Tasks (>50ms)
    if ('PerformanceObserver' in window) {
      const longTaskObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.duration > 50) {
            console.warn(`Long task detected: ${entry.duration}ms`);
            this.onLongTask?.(entry.duration);
          }
        }
      });

      try {
        longTaskObserver.observe({ entryTypes: ['longtask'] });
        this.observers.push(longTaskObserver);
      } catch (e) {
        console.warn('Long task monitoring not supported');
      }

      // Monitor Layout Shifts
      const layoutShiftObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if ((entry as any).value > 0.1) {
            console.warn(`Layout shift detected: ${(entry as any).value}`);
            this.onLayoutShift?.((entry as any).value);
          }
        }
      });

      try {
        layoutShiftObserver.observe({ entryTypes: ['layout-shift'] });
        this.observers.push(layoutShiftObserver);
      } catch (e) {
        console.warn('Layout shift monitoring not supported');
      }

      // Monitor First Input Delay
      const fidObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          const fid = entry.processingStart - entry.startTime;
          this.metrics.interactionLatency = fid;
          if (fid > 100) {
            console.warn(`High input delay: ${fid}ms`);
            this.onHighInputDelay?.(fid);
          }
        }
      });

      try {
        fidObserver.observe({ entryTypes: ['first-input'] });
        this.observers.push(fidObserver);
      } catch (e) {
        console.warn('First input delay monitoring not supported');
      }
    }
  }

  private startFPSMonitoring() {
    const measureFPS = (timestamp: number) => {
      if (this.lastFrameTime) {
        const delta = timestamp - this.lastFrameTime;
        const fps = 1000 / delta;
        
        this.fpsHistory.push(fps);
        if (this.fpsHistory.length > 60) {
          this.fpsHistory.shift();
        }
        
        this.metrics.fps = this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
        
        if (this.metrics.fps < 30) {
          this.onLowFPS?.(this.metrics.fps);
        }
      }
      
      this.lastFrameTime = timestamp;
      this.frameCount++;
      
      requestAnimationFrame(measureFPS);
    };
    
    requestAnimationFrame(measureFPS);
  }

  private monitorMemoryUsage() {
    if ('memory' in performance) {
      setInterval(() => {
        const memory = (performance as any).memory;
        this.metrics.memoryUsage = memory.usedJSHeapSize / memory.jsHeapSizeLimit;
        
        if (this.metrics.memoryUsage > 0.8) {
          console.warn(`High memory usage: ${(this.metrics.memoryUsage * 100).toFixed(1)}%`);
          this.onHighMemoryUsage?.(this.metrics.memoryUsage);
        }
      }, 5000);
    }
  }

  // Measure interaction latency
  measureInteraction<T>(fn: () => T): T {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    
    const latency = end - start;
    this.metrics.interactionLatency = latency;
    
    if (latency > 100) {
      console.warn(`Slow interaction: ${latency}ms`);
      this.onSlowInteraction?.(latency);
    }
    
    return result;
  }

  // Measure async interaction latency
  async measureAsyncInteraction<T>(fn: () => Promise<T>): Promise<T> {
    const start = performance.now();
    const result = await fn();
    const end = performance.now();
    
    const latency = end - start;
    this.metrics.interactionLatency = latency;
    
    if (latency > 100) {
      console.warn(`Slow async interaction: ${latency}ms`);
      this.onSlowInteraction?.(latency);
    }
    
    return result;
  }

  // Measure render time
  measureRender(componentName: string, renderFn: () => void) {
    const start = performance.now();
    renderFn();
    const end = performance.now();
    
    const renderTime = end - start;
    this.metrics.renderTime = renderTime;
    
    if (renderTime > 16) { // 60fps = 16.67ms per frame
      console.warn(`Slow render (${componentName}): ${renderTime}ms`);
      this.onSlowRender?.(componentName, renderTime);
    }
  }

  // Measure network latency
  async measureNetworkLatency(url: string): Promise<number> {
    const start = performance.now();
    
    try {
      await fetch(url, { method: 'HEAD' });
      const end = performance.now();
      const latency = end - start;
      
      this.metrics.networkLatency = latency;
      return latency;
    } catch (error) {
      console.error('Network latency measurement failed:', error);
      return -1;
    }
  }

  // Get current metrics
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  // Get performance score (0-100)
  getPerformanceScore(): number {
    let score = 100;
    
    // Deduct points for poor metrics
    if (this.metrics.interactionLatency > 100) {
      score -= Math.min(30, (this.metrics.interactionLatency - 100) / 10);
    }
    
    if (this.metrics.fps < 60) {
      score -= Math.min(20, (60 - this.metrics.fps) / 2);
    }
    
    if (this.metrics.memoryUsage > 0.7) {
      score -= Math.min(25, (this.metrics.memoryUsage - 0.7) * 100);
    }
    
    if (this.metrics.networkLatency > 200) {
      score -= Math.min(15, (this.metrics.networkLatency - 200) / 20);
    }
    
    return Math.max(0, Math.round(score));
  }

  // Get recommendations for performance improvements
  getRecommendations(): string[] {
    const recommendations: string[] = [];
    
    if (this.metrics.interactionLatency > 100) {
      recommendations.push('Consider debouncing user interactions or optimizing event handlers');
    }
    
    if (this.metrics.fps < 30) {
      recommendations.push('Reduce complex animations or use CSS transforms for better performance');
    }
    
    if (this.metrics.memoryUsage > 0.8) {
      recommendations.push('Check for memory leaks and optimize component cleanup');
    }
    
    if (this.metrics.networkLatency > 500) {
      recommendations.push('Consider implementing request caching or using a CDN');
    }
    
    return recommendations;
  }

  // Event callbacks
  onLongTask?: (duration: number) => void;
  onLayoutShift?: (value: number) => void;
  onHighInputDelay?: (delay: number) => void;
  onLowFPS?: (fps: number) => void;
  onHighMemoryUsage?: (usage: number) => void;
  onSlowInteraction?: (latency: number) => void;
  onSlowRender?: (component: string, time: number) => void;

  // Cleanup
  destroy() {
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
  }
}

// React hook for performance monitoring
import { useEffect, useRef, useState } from 'react';

export function usePerformanceMonitor() {
  const monitor = useRef<PerformanceMonitor | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    interactionLatency: 0,
    renderTime: 0,
    memoryUsage: 0,
    networkLatency: 0,
    fps: 0,
    bundleSize: 0
  });
  const [score, setScore] = useState(100);

  useEffect(() => {
    monitor.current = new PerformanceMonitor();
    
    // Update metrics every second
    const interval = setInterval(() => {
      if (monitor.current) {
        setMetrics(monitor.current.getMetrics());
        setScore(monitor.current.getPerformanceScore());
      }
    }, 1000);

    return () => {
      clearInterval(interval);
      monitor.current?.destroy();
    };
  }, []);

  const measureInteraction = <T,>(fn: () => T): T => {
    return monitor.current?.measureInteraction(fn) || fn();
  };

  const measureAsyncInteraction = async <T,>(fn: () => Promise<T>): Promise<T> => {
    return monitor.current?.measureAsyncInteraction(fn) || fn();
  };

  const measureRender = (componentName: string, renderFn: () => void) => {
    monitor.current?.measureRender(componentName, renderFn);
  };

  const getRecommendations = (): string[] => {
    return monitor.current?.getRecommendations() || [];
  };

  return {
    metrics,
    score,
    measureInteraction,
    measureAsyncInteraction,
    measureRender,
    getRecommendations
  };
}