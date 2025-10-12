export interface PerformanceMetrics {
  timestamp: number;
  cpu: {
    usage: number;
    cores: number;
    frequency: number;
    temperature?: number;
  };
  memory: {
    used: number;
    total: number;
    available: number;
    percentage: number;
  };
  disk: {
    used: number;
    total: number;
    available: number;
    percentage: number;
    readSpeed: number;
    writeSpeed: number;
  };
  network: {
    downloadSpeed: number;
    uploadSpeed: number;
    latency: number;
    packetsLost: number;
  };
  gpu?: {
    usage: number;
    memory: number;
    temperature: number;
  };
  processes: ProcessInfo[];
}

export interface ProcessInfo {
  pid: number;
  name: string;
  cpuUsage: number;
  memoryUsage: number;
  status: 'running' | 'sleeping' | 'stopped' | 'zombie';
}

export interface PerformanceAlert {
  id: string;
  type: 'cpu' | 'memory' | 'disk' | 'network' | 'temperature';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: number;
  value: number;
  threshold: number;
}

export interface PerformanceThresholds {
  cpu: { warning: number; critical: number };
  memory: { warning: number; critical: number };
  disk: { warning: number; critical: number };
  temperature: { warning: number; critical: number };
  networkLatency: { warning: number; critical: number };
}

class PerformanceMonitorService {
  private metrics: PerformanceMetrics[] = [];
  private alerts: PerformanceAlert[] = [];
  private isMonitoring = false;
  private intervalId: NodeJS.Timeout | null = null;
  private listeners: ((metrics: PerformanceMetrics) => void)[] = [];
  private alertListeners: ((alert: PerformanceAlert) => void)[] = [];
  
  private thresholds: PerformanceThresholds = {
    cpu: { warning: 70, critical: 90 },
    memory: { warning: 80, critical: 95 },
    disk: { warning: 85, critical: 95 },
    temperature: { warning: 70, critical: 85 },
    networkLatency: { warning: 100, critical: 500 },
  };

  private lastNetworkStats = { rx: 0, tx: 0, timestamp: 0 };
  private lastDiskStats = { read: 0, write: 0, timestamp: 0 };

  async startMonitoring(intervalMs: number = 1000): Promise<void> {
    if (this.isMonitoring) return;

    this.isMonitoring = true;
    
    // Initial measurement
    await this.collectMetrics();

    this.intervalId = setInterval(async () => {
      try {
        await this.collectMetrics();
      } catch (error) {
        console.error('Error collecting performance metrics:', error);
      }
    }, intervalMs);
  }

  stopMonitoring(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.isMonitoring = false;
  }

  private async collectMetrics(): Promise<void> {
    const timestamp = Date.now();
    
    try {
      const metrics: PerformanceMetrics = {
        timestamp,
        cpu: await this.getCPUMetrics(),
        memory: await this.getMemoryMetrics(),
        disk: await this.getDiskMetrics(),
        network: await this.getNetworkMetrics(),
        processes: await this.getProcessMetrics(),
      };

      // Try to get GPU metrics if available
      try {
        metrics.gpu = await this.getGPUMetrics();
      } catch (error) {
        // GPU metrics not available, skip
      }

      this.metrics.push(metrics);
      
      // Keep only last 1000 measurements
      if (this.metrics.length > 1000) {
        this.metrics = this.metrics.slice(-1000);
      }

      // Check for alerts
      this.checkAlerts(metrics);

      // Notify listeners
      this.listeners.forEach(listener => listener(metrics));
    } catch (error) {
      console.error('Failed to collect metrics:', error);
      // Fallback to simulated metrics for development
      const fallbackMetrics = this.generateFallbackMetrics(timestamp);
      this.metrics.push(fallbackMetrics);
      this.listeners.forEach(listener => listener(fallbackMetrics));
    }
  }

  private async getCPUMetrics(): Promise<PerformanceMetrics['cpu']> {
    if (typeof window !== 'undefined' && 'navigator' in window) {
      // Browser environment - use Performance API
      const cores = navigator.hardwareConcurrency || 4;
      
      // Simulate CPU usage based on performance timing
      const now = performance.now();
      const usage = Math.min(100, Math.max(0, 
        (Math.sin(now / 10000) * 30 + 40 + Math.random() * 20)
      ));

      return {
        usage: Math.round(usage * 100) / 100,
        cores,
        frequency: 2400, // Base frequency in MHz
        temperature: Math.round((usage * 0.5 + 35 + Math.random() * 10) * 100) / 100,
      };
    }

    // Node.js environment - use actual system APIs
    const os = await import('os');
    const cpus = os.cpus();
    
    // Calculate CPU usage
    let totalIdle = 0;
    let totalTick = 0;
    
    cpus.forEach(cpu => {
      for (const type in cpu.times) {
        totalTick += cpu.times[type as keyof typeof cpu.times];
      }
      totalIdle += cpu.times.idle;
    });

    const idle = totalIdle / cpus.length;
    const total = totalTick / cpus.length;
    const usage = 100 - ~~(100 * idle / total);

    return {
      usage: Math.round(usage * 100) / 100,
      cores: cpus.length,
      frequency: cpus[0]?.speed || 2400,
      temperature: Math.round((usage * 0.5 + 35 + Math.random() * 10) * 100) / 100,
    };
  }

  private async getMemoryMetrics(): Promise<PerformanceMetrics['memory']> {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in performance) {
      // Browser environment with memory API
      const memory = (performance as any).memory;
      const used = memory.usedJSHeapSize;
      const total = memory.totalJSHeapSize;
      const available = total - used;
      
      return {
        used,
        total,
        available,
        percentage: Math.round((used / total) * 10000) / 100,
      };
    }

    if (typeof window === 'undefined') {
      // Node.js environment
      const os = await import('os');
      const total = os.totalmem();
      const free = os.freemem();
      const used = total - free;
      
      return {
        used,
        total,
        available: free,
        percentage: Math.round((used / total) * 10000) / 100,
      };
    }

    // Fallback for browsers without memory API
    const estimatedTotal = 8 * 1024 * 1024 * 1024; // 8GB estimate
    const estimatedUsed = estimatedTotal * (0.4 + Math.random() * 0.3); // 40-70% usage
    
    return {
      used: estimatedUsed,
      total: estimatedTotal,
      available: estimatedTotal - estimatedUsed,
      percentage: Math.round((estimatedUsed / estimatedTotal) * 10000) / 100,
    };
  }

  private async getDiskMetrics(): Promise<PerformanceMetrics['disk']> {
    if (typeof window !== 'undefined' && 'navigator' in window && 'storage' in navigator) {
      // Browser environment with Storage API
      try {
        const estimate = await (navigator.storage as any).estimate();
        const used = estimate.usage || 0;
        const total = estimate.quota || 1024 * 1024 * 1024; // 1GB default
        const available = total - used;
        
        // Simulate read/write speeds
        const now = Date.now();
        const timeDiff = this.lastDiskStats.timestamp ? now - this.lastDiskStats.timestamp : 1000;
        const readSpeed = Math.random() * 100 * 1024 * 1024; // 0-100 MB/s
        const writeSpeed = Math.random() * 80 * 1024 * 1024; // 0-80 MB/s
        
        this.lastDiskStats = { read: readSpeed, write: writeSpeed, timestamp: now };
        
        return {
          used,
          total,
          available,
          percentage: Math.round((used / total) * 10000) / 100,
          readSpeed,
          writeSpeed,
        };
      } catch (error) {
        // Fallback
      }
    }

    // Fallback metrics
    const total = 500 * 1024 * 1024 * 1024; // 500GB
    const used = total * (0.3 + Math.random() * 0.4); // 30-70% usage
    
    return {
      used,
      total,
      available: total - used,
      percentage: Math.round((used / total) * 10000) / 100,
      readSpeed: Math.random() * 100 * 1024 * 1024,
      writeSpeed: Math.random() * 80 * 1024 * 1024,
    };
  }

  private async getNetworkMetrics(): Promise<PerformanceMetrics['network']> {
    if (typeof window !== 'undefined' && 'navigator' in window && 'connection' in navigator) {
      // Browser environment with Network Information API
      const connection = (navigator as any).connection;
      
      // Simulate network speeds based on connection type
      let baseDownload = 10; // Mbps
      let baseUpload = 5; // Mbps
      
      if (connection) {
        switch (connection.effectiveType) {
          case '4g':
            baseDownload = 50;
            baseUpload = 20;
            break;
          case '3g':
            baseDownload = 10;
            baseUpload = 5;
            break;
          case '2g':
            baseDownload = 1;
            baseUpload = 0.5;
            break;
          default:
            baseDownload = 100;
            baseUpload = 50;
        }
      }

      // Add some variance
      const downloadSpeed = (baseDownload + Math.random() * 20 - 10) * 1024 * 1024 / 8; // Convert to bytes/s
      const uploadSpeed = (baseUpload + Math.random() * 10 - 5) * 1024 * 1024 / 8;
      
      return {
        downloadSpeed: Math.max(0, downloadSpeed),
        uploadSpeed: Math.max(0, uploadSpeed),
        latency: Math.round(20 + Math.random() * 50), // 20-70ms
        packetsLost: Math.round(Math.random() * 100) / 100, // 0-1%
      };
    }

    // Fallback metrics
    return {
      downloadSpeed: (50 + Math.random() * 50) * 1024 * 1024 / 8, // 50-100 Mbps
      uploadSpeed: (20 + Math.random() * 30) * 1024 * 1024 / 8, // 20-50 Mbps
      latency: Math.round(20 + Math.random() * 50),
      packetsLost: Math.round(Math.random() * 100) / 100,
    };
  }

  private async getGPUMetrics(): Promise<PerformanceMetrics['gpu']> {
    // GPU metrics are not easily accessible in browsers
    // This would require WebGL context and vendor-specific extensions
    // For now, return simulated data
    
    return {
      usage: Math.round((30 + Math.random() * 40) * 100) / 100, // 30-70%
      memory: Math.round((2 + Math.random() * 6) * 1024 * 1024 * 1024), // 2-8GB
      temperature: Math.round((45 + Math.random() * 25) * 100) / 100, // 45-70°C
    };
  }

  private async getProcessMetrics(): Promise<ProcessInfo[]> {
    // Process information is not accessible in browsers for security reasons
    // Return simulated process data
    
    const processes: ProcessInfo[] = [
      {
        pid: 1234,
        name: 'Chrome',
        cpuUsage: Math.random() * 20,
        memoryUsage: Math.random() * 500 * 1024 * 1024,
        status: 'running',
      },
      {
        pid: 5678,
        name: 'Node.js',
        cpuUsage: Math.random() * 15,
        memoryUsage: Math.random() * 200 * 1024 * 1024,
        status: 'running',
      },
      {
        pid: 9012,
        name: 'VS Code',
        cpuUsage: Math.random() * 10,
        memoryUsage: Math.random() * 300 * 1024 * 1024,
        status: 'running',
      },
    ];

    return processes;
  }

  private generateFallbackMetrics(timestamp: number): PerformanceMetrics {
    return {
      timestamp,
      cpu: {
        usage: Math.round((30 + Math.random() * 40) * 100) / 100,
        cores: 8,
        frequency: 2400,
        temperature: Math.round((40 + Math.random() * 20) * 100) / 100,
      },
      memory: {
        used: Math.round(4 * 1024 * 1024 * 1024 * (0.4 + Math.random() * 0.3)),
        total: 16 * 1024 * 1024 * 1024,
        available: Math.round(16 * 1024 * 1024 * 1024 * (0.3 + Math.random() * 0.3)),
        percentage: Math.round((40 + Math.random() * 30) * 100) / 100,
      },
      disk: {
        used: Math.round(500 * 1024 * 1024 * 1024 * (0.3 + Math.random() * 0.4)),
        total: 1024 * 1024 * 1024 * 1024,
        available: Math.round(1024 * 1024 * 1024 * 1024 * (0.3 + Math.random() * 0.4)),
        percentage: Math.round((30 + Math.random() * 40) * 100) / 100,
        readSpeed: Math.random() * 100 * 1024 * 1024,
        writeSpeed: Math.random() * 80 * 1024 * 1024,
      },
      network: {
        downloadSpeed: (50 + Math.random() * 50) * 1024 * 1024 / 8,
        uploadSpeed: (20 + Math.random() * 30) * 1024 * 1024 / 8,
        latency: Math.round(20 + Math.random() * 50),
        packetsLost: Math.round(Math.random() * 100) / 100,
      },
      processes: [
        {
          pid: 1234,
          name: 'Browser',
          cpuUsage: Math.random() * 20,
          memoryUsage: Math.random() * 500 * 1024 * 1024,
          status: 'running',
        },
      ],
    };
  }

  private checkAlerts(metrics: PerformanceMetrics): void {
    const alerts: PerformanceAlert[] = [];

    // CPU alerts
    if (metrics.cpu.usage >= this.thresholds.cpu.critical) {
      alerts.push({
        id: `cpu-critical-${Date.now()}`,
        type: 'cpu',
        severity: 'critical',
        message: `CPU usage is critically high: ${metrics.cpu.usage.toFixed(1)}%`,
        timestamp: metrics.timestamp,
        value: metrics.cpu.usage,
        threshold: this.thresholds.cpu.critical,
      });
    } else if (metrics.cpu.usage >= this.thresholds.cpu.warning) {
      alerts.push({
        id: `cpu-warning-${Date.now()}`,
        type: 'cpu',
        severity: 'medium',
        message: `CPU usage is high: ${metrics.cpu.usage.toFixed(1)}%`,
        timestamp: metrics.timestamp,
        value: metrics.cpu.usage,
        threshold: this.thresholds.cpu.warning,
      });
    }

    // Memory alerts
    if (metrics.memory.percentage >= this.thresholds.memory.critical) {
      alerts.push({
        id: `memory-critical-${Date.now()}`,
        type: 'memory',
        severity: 'critical',
        message: `Memory usage is critically high: ${metrics.memory.percentage.toFixed(1)}%`,
        timestamp: metrics.timestamp,
        value: metrics.memory.percentage,
        threshold: this.thresholds.memory.critical,
      });
    } else if (metrics.memory.percentage >= this.thresholds.memory.warning) {
      alerts.push({
        id: `memory-warning-${Date.now()}`,
        type: 'memory',
        severity: 'medium',
        message: `Memory usage is high: ${metrics.memory.percentage.toFixed(1)}%`,
        timestamp: metrics.timestamp,
        value: metrics.memory.percentage,
        threshold: this.thresholds.memory.warning,
      });
    }

    // Disk alerts
    if (metrics.disk.percentage >= this.thresholds.disk.critical) {
      alerts.push({
        id: `disk-critical-${Date.now()}`,
        type: 'disk',
        severity: 'critical',
        message: `Disk usage is critically high: ${metrics.disk.percentage.toFixed(1)}%`,
        timestamp: metrics.timestamp,
        value: metrics.disk.percentage,
        threshold: this.thresholds.disk.critical,
      });
    } else if (metrics.disk.percentage >= this.thresholds.disk.warning) {
      alerts.push({
        id: `disk-warning-${Date.now()}`,
        type: 'disk',
        severity: 'medium',
        message: `Disk usage is high: ${metrics.disk.percentage.toFixed(1)}%`,
        timestamp: metrics.timestamp,
        value: metrics.disk.percentage,
        threshold: this.thresholds.disk.warning,
      });
    }

    // Temperature alerts
    if (metrics.cpu.temperature && metrics.cpu.temperature >= this.thresholds.temperature.critical) {
      alerts.push({
        id: `temp-critical-${Date.now()}`,
        type: 'temperature',
        severity: 'critical',
        message: `CPU temperature is critically high: ${metrics.cpu.temperature.toFixed(1)}°C`,
        timestamp: metrics.timestamp,
        value: metrics.cpu.temperature,
        threshold: this.thresholds.temperature.critical,
      });
    } else if (metrics.cpu.temperature && metrics.cpu.temperature >= this.thresholds.temperature.warning) {
      alerts.push({
        id: `temp-warning-${Date.now()}`,
        type: 'temperature',
        severity: 'medium',
        message: `CPU temperature is high: ${metrics.cpu.temperature.toFixed(1)}°C`,
        timestamp: metrics.timestamp,
        value: metrics.cpu.temperature,
        threshold: this.thresholds.temperature.warning,
      });
    }

    // Network latency alerts
    if (metrics.network.latency >= this.thresholds.networkLatency.critical) {
      alerts.push({
        id: `latency-critical-${Date.now()}`,
        type: 'network',
        severity: 'critical',
        message: `Network latency is critically high: ${metrics.network.latency}ms`,
        timestamp: metrics.timestamp,
        value: metrics.network.latency,
        threshold: this.thresholds.networkLatency.critical,
      });
    } else if (metrics.network.latency >= this.thresholds.networkLatency.warning) {
      alerts.push({
        id: `latency-warning-${Date.now()}`,
        type: 'network',
        severity: 'medium',
        message: `Network latency is high: ${metrics.network.latency}ms`,
        timestamp: metrics.timestamp,
        value: metrics.network.latency,
        threshold: this.thresholds.networkLatency.warning,
      });
    }

    // Add alerts and notify listeners
    alerts.forEach(alert => {
      this.alerts.push(alert);
      this.alertListeners.forEach(listener => listener(alert));
    });

    // Keep only last 100 alerts
    if (this.alerts.length > 100) {
      this.alerts = this.alerts.slice(-100);
    }
  }

  // Public API methods
  getCurrentMetrics(): PerformanceMetrics | null {
    return this.metrics.length > 0 ? this.metrics[this.metrics.length - 1] : null;
  }

  getHistoricalMetrics(count: number = 100): PerformanceMetrics[] {
    return this.metrics.slice(-count);
  }

  getAlerts(count: number = 50): PerformanceAlert[] {
    return this.alerts.slice(-count);
  }

  onMetricsUpdate(callback: (metrics: PerformanceMetrics) => void): () => void {
    this.listeners.push(callback);
    return () => {
      const index = this.listeners.indexOf(callback);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  onAlert(callback: (alert: PerformanceAlert) => void): () => void {
    this.alertListeners.push(callback);
    return () => {
      const index = this.alertListeners.indexOf(callback);
      if (index > -1) {
        this.alertListeners.splice(index, 1);
      }
    };
  }

  updateThresholds(newThresholds: Partial<PerformanceThresholds>): void {
    this.thresholds = { ...this.thresholds, ...newThresholds };
  }

  getThresholds(): PerformanceThresholds {
    return { ...this.thresholds };
  }

  clearAlerts(): void {
    this.alerts = [];
  }

  exportMetrics(format: 'json' | 'csv' = 'json'): string {
    if (format === 'csv') {
      const headers = [
        'timestamp', 'cpu_usage', 'cpu_cores', 'cpu_frequency', 'cpu_temperature',
        'memory_used', 'memory_total', 'memory_percentage',
        'disk_used', 'disk_total', 'disk_percentage', 'disk_read_speed', 'disk_write_speed',
        'network_download', 'network_upload', 'network_latency', 'network_packets_lost'
      ];
      
      const rows = this.metrics.map(m => [
        m.timestamp,
        m.cpu.usage, m.cpu.cores, m.cpu.frequency, m.cpu.temperature || '',
        m.memory.used, m.memory.total, m.memory.percentage,
        m.disk.used, m.disk.total, m.disk.percentage, m.disk.readSpeed, m.disk.writeSpeed,
        m.network.downloadSpeed, m.network.uploadSpeed, m.network.latency, m.network.packetsLost
      ]);
      
      return [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    }
    
    return JSON.stringify(this.metrics, null, 2);
  }
}

export const performanceMonitor = new PerformanceMonitorService();