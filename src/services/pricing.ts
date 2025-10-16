import { PricingPlan, PlanType } from '@/types/saas';

export const PRICING_PLANS: PricingPlan[] = [
  {
    id: 'free',
    name: 'Free',
    price: 0,
    interval: 'month',
    features: [
      '5 workflows',
      '100 executions/month',
      '100 MB storage',
      '1,000 API calls/month',
      '1 team member',
      'Community support',
      'Basic analytics'
    ],
    limits: {
      workflows: 5,
      executions: 100,
      storage: 100,
      apiCalls: 1000,
      teamMembers: 1,
      customNodes: false,
      prioritySupport: false,
      advancedAnalytics: false
    }
  },
  {
    id: 'pro',
    name: 'Pro',
    price: 29,
    interval: 'month',
    popular: true,
    features: [
      'Unlimited workflows',
      '10,000 executions/month',
      '10 GB storage',
      '100,000 API calls/month',
      '10 team members',
      'Custom nodes',
      'Priority support',
      'Advanced analytics',
      'Webhook triggers',
      'API access'
    ],
    limits: {
      workflows: -1, // unlimited
      executions: 10000,
      storage: 10240,
      apiCalls: 100000,
      teamMembers: 10,
      customNodes: true,
      prioritySupport: true,
      advancedAnalytics: true
    }
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    price: 99,
    interval: 'month',
    features: [
      'Unlimited everything',
      'Custom execution limits',
      'Unlimited storage',
      'Unlimited API calls',
      'Unlimited team members',
      'Custom nodes & integrations',
      'Dedicated support',
      'Advanced analytics & reporting',
      'SSO & SAML',
      'Custom SLA',
      'On-premise deployment',
      'White-label options'
    ],
    limits: {
      workflows: -1,
      executions: -1,
      storage: -1,
      apiCalls: -1,
      teamMembers: -1,
      customNodes: true,
      prioritySupport: true,
      advancedAnalytics: true
    }
  }
];

export function getPlanByType(planType: PlanType): PricingPlan | undefined {
  return PRICING_PLANS.find(plan => plan.id === planType);
}

export function canUpgrade(currentPlan: PlanType, targetPlan: PlanType): boolean {
  const planOrder: PlanType[] = ['free', 'pro', 'enterprise'];
  return planOrder.indexOf(targetPlan) > planOrder.indexOf(currentPlan);
}

export function checkUsageLimit(
  usage: number,
  limit: number
): { allowed: boolean; percentage: number } {
  if (limit === -1) return { allowed: true, percentage: 0 };
  const percentage = (usage / limit) * 100;
  return { allowed: usage < limit, percentage };
}
