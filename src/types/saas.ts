// SaaS Types and Interfaces

export type PlanType = 'free' | 'pro' | 'enterprise';

export interface User {
    id: string;
    email: string;
    name: string;
    avatar?: string;
    role: 'owner' | 'admin' | 'member';
    createdAt: Date;
    lastLogin?: Date;
}

export interface Workspace {
    id: string;
    name: string;
    slug: string;
    ownerId: string;
    plan: PlanType;
    members: WorkspaceMember[];
    settings: WorkspaceSettings;
    createdAt: Date;
    updatedAt: Date;
}

export interface WorkspaceMember {
    userId: string;
    role: 'owner' | 'admin' | 'member';
    joinedAt: Date;
}

export interface WorkspaceSettings {
    allowInvites: boolean;
    maxMembers: number;
    customDomain?: string;
}

export interface Subscription {
    id: string;
    workspaceId: string;
    plan: PlanType;
    status: 'active' | 'canceled' | 'past_due' | 'trialing';
    currentPeriodStart: Date;
    currentPeriodEnd: Date;
    cancelAtPeriodEnd: boolean;
    stripeCustomerId?: string;
    stripeSubscriptionId?: string;
}

export interface PricingPlan {
    id: PlanType;
    name: string;
    price: number;
    interval: 'month' | 'year';
    features: string[];
    limits: UsageLimits;
    popular?: boolean;
}

export interface UsageLimits {
    workflows: number;
    executions: number;
    storage: number; // in MB
    apiCalls: number;
    teamMembers: number;
    customNodes: boolean;
    prioritySupport: boolean;
    advancedAnalytics: boolean;
}

export interface Usage {
    workspaceId: string;
    period: string; // YYYY-MM
    workflows: number;
    executions: number;
    storage: number;
    apiCalls: number;
}

export interface ApiKey {
    id: string;
    name: string;
    key: string;
    workspaceId: string;
    createdBy: string;
    createdAt: Date;
    lastUsed?: Date;
    expiresAt?: Date;
}
