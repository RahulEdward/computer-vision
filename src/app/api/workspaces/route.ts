import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import prisma from '@/lib/prisma';
import { z } from 'zod';

const createWorkspaceSchema = z.object({
  name: z.string().min(2),
  slug: z.string().optional()
});

// GET /api/workspaces - Get user's workspaces
export async function GET(req: NextRequest) {
  try {
    const session = await getServerSession(authOptions);
    const userId = (session?.user as any)?.id;
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const workspaces = await prisma.workspace.findMany({
      where: {
        members: {
          some: {
            userId: userId
          }
        }
      },
      include: {
        members: {
          include: {
            user: {
              select: {
                id: true,
                name: true,
                email: true,
                avatar: true
              }
            }
          }
        },
        subscription: true,
        _count: {
          select: {
            workflows: true,
            members: true
          }
        }
      }
    });

    return NextResponse.json({ workspaces });
  } catch (error) {
    console.error('Get workspaces error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST /api/workspaces - Create new workspace
export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession(authOptions);
    const userId = (session?.user as any)?.id;
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    const { name, slug } = createWorkspaceSchema.parse(body);

    const workspaceSlug = slug || `${name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;

    const workspace = await prisma.workspace.create({
      data: {
        name,
        slug: workspaceSlug,
        ownerId: userId,
        plan: 'free',
        members: {
          create: {
            userId: userId,
            role: 'owner'
          }
        },
        subscription: {
          create: {
            plan: 'free',
            status: 'active',
            currentPeriodEnd: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000)
          }
        }
      },
      include: {
        members: true,
        subscription: true
      }
    });

    return NextResponse.json({ workspace });
  } catch (error) {
    console.error('Create workspace error:', error);
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Invalid input', details: error.errors },
        { status: 400 }
      );
    }
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
