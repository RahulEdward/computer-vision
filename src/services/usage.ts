import { Usage, UsageLimits } from '@/types/saas';
import { getPlanByType } from './pricing';

class UsageService {
  private usage: Map<string, Usage> = new Map();

  async trackExecution(workspaceId: string): Promise<boolean> {
    const period = this.getCurrentPeriod();
    const key = `${workspaceId}-${period}`;
    
    const current = this.usage.get(key) || this.createUsage(workspaceId, period);
    current.executions++;
    this.usage.set(key, current);
    
    return true;
  }

  async trackApiCall(workspaceId: string): Promise<boolean> {
    const period = this.getCurrentPeriod();
    const key = `${workspaceId}-${period}`;
    
    const current = this.usage.get(key) || this.createUsage(workspaceId, period);
    current.apiCalls++;
    this.usage.set(key, current);
    
    return true;
  }

  async getUsage(workspaceId: string, period?: string): Promise<Usage> {
    const targetPeriod = period || this.getCurrentPeriod();
    const key = `${workspaceId}-${targetPeriod}`;
    return this.usage.get(key) || this.createUsage(workspaceId, targetPeriod);
  }

  async checkLimit(
    workspaceId: string,
    planType: string,
    limitType: keyof UsageLimits
  ): Promise<{ allowed: boolean; current: number; limit: number }> {
    const plan = getPlanByType(planType as any);
    if (!plan) return { allowed: false, current: 0, limit: 0 };

    const usage = await this.getUsage(workspaceId);
    const limit = plan.limits[limitType] as number;
    
    if (limit === -1) {
      return { allowed: true, current: usage[limitType as keyof Usage] as number, limit: -1 };
    }

    const current = usage[limitType as keyof Usage] as number;
    return { allowed: current < limit, current, limit };
  }

  private getCurrentPeriod(): string {
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
  }

  private createUsage(workspaceId: string, period: string): Usage {
    return {
      workspaceId,
      period,
      workflows: 0,
      executions: 0,
      storage: 0,
      apiCalls: 0
    };
  }
}

export const usageService = new UsageService();
