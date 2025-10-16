import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import prisma from '@/lib/prisma';
import { z } from 'zod';

const createWorkflowSchema = z.object({
    workspaceId: z.string(),
    name: z.string().min(1),
    description: z.string().optional(),
    nodes: z.any(),
    edges: z.any()
});

// GET /api/workflows - Get workflows
export async function GET(req: NextRequest) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user?.id) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const { searchParams } = new URL(req.url);
        const workspaceId = searchParams.get('workspaceId');

        if (!workspaceId) {
            return NextResponse.json({ error: 'Workspace ID required' }, { status: 400 });
        }

        // Check access
        const member = await prisma.workspaceMember.findFirst({
            where: {
                workspaceId,
                userId: session.user.id
            }
        });

        if (!member) {
            return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
        }

        const workflows = await prisma.workflow.findMany({
            where: { workspaceId },
            orderBy: { updatedAt: 'desc' }
        });

        return NextResponse.json({ workflows });
    } catch (error) {
        console.error('Get workflows error:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}

// POST /api/workflows - Create workflow
export async function POST(req: NextRequest) {
    try {
        const session = await getServerSession(authOptions);
        if (!session?.user?.id) {
            return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
        }

        const body = await req.json();
        const { workspaceId, name, description, nodes, edges } = createWorkflowSchema.parse(body);

        // Check access
        const member = await prisma.workspaceMember.findFirst({
            where: {
                workspaceId,
                userId: session.user.id
            }
        });

        if (!member) {
            return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
        }

        // Check workflow limit
        const workspace = await prisma.workspace.findUnique({
            where: { id: workspaceId },
            include: {
                subscription: true,
                _count: { select: { workflows: true } }
            }
        });

        const plan = workspace?.subscription?.plan || 'free';
        const workflowCount = workspace?._count.workflows || 0;

        // Free plan: 5 workflows limit
        if (plan === 'free' && workflowCount >= 5) {
            return NextResponse.json(
                { error: 'Workflow limit reached. Upgrade to Pro for unlimited workflows.' },
                { status: 403 }
            );
        }

        const workflow = await prisma.workflow.create({
            data: {
                workspaceId,
                name,
                description,
                nodes,
                edges,
                createdBy: session.user.id,
                status: 'draft'
            }
        });

        // Update usage
        const now = new Date();
        const period = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;

        await prisma.usage.upsert({
            where: {
                workspaceId_period: { workspaceId, period }
            },
            update: {
                workflows: { increment: 1 }
            },
            create: {
                workspaceId,
                period,
                workflows: 1
            }
        });

        return NextResponse.json({ workflow });
    } catch (error) {
        console.error('Create workflow error:', error);
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
