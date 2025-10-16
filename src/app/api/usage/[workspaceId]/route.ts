import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import prisma from '@/lib/prisma';

// GET /api/usage/[workspaceId] - Get workspace usage
export async function GET(
  req: NextRequest,
  { params }: { params: { workspaceId: string } }
) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user?.id) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { workspaceId } = params;

    // Check if user has access to workspace
    const member = await prisma.workspaceMember.findFirst({
      where: {
        workspaceId,
        userId: session.user.id
      }
    });

    if (!member) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    // Get current period
    const now = new Date();
    const period = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;

    // Get or create usage record
    let usage = await prisma.usage.findUnique({
      where: {
        workspaceId_period: {
          workspaceId,
          period
        }
      }
    });

    if (!usage) {
      usage = await prisma.usage.create({
        data: {
          workspaceId,
          period,
          workflows: 0,
          executions: 0,
          storage: 0,
          apiCalls: 0
        }
      });
    }

    // Get workspace with subscription
    const workspace = await prisma.workspace.findUnique({
      where: { id: workspaceId },
      include: {
        subscription: true
      }
    });

    return NextResponse.json({
      usage,
      plan: workspace?.subscription?.plan || 'free',
      period
    });
  } catch (error) {
    console.error('Get usage error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST /api/usage/[workspaceId] - Track usage
export async function POST(
  req: NextRequest,
  { params }: { params: { workspaceId: string } }
) {
  try {
    const session = await getServerSession(authOptions);
    if (!session?.user?.id) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { workspaceId } = params;
    const body = await req.json();
    const { type, amount = 1 } = body;

    // Check if user has access
    const member = await prisma.workspaceMember.findFirst({
      where: {
        workspaceId,
        userId: session.user.id
      }
    });

    if (!member) {
      return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
    }

    const now = new Date();
    const period = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;

    // Update usage
    const usage = await prisma.usage.upsert({
      where: {
        workspaceId_period: {
          workspaceId,
          period
        }
      },
      update: {
        [type]: {
          increment: amount
        }
      },
      create: {
        workspaceId,
        period,
        [type]: amount
      }
    });

    return NextResponse.json({ usage });
  } catch (error) {
    console.error('Track usage error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
