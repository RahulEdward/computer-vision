import { NextRequest, NextResponse } from 'next/server';
// @ts-ignore
import bcrypt from 'bcryptjs';
import prisma from '@/lib/prisma';
import { z } from 'zod';

const signupSchema = z.object({
  name: z.string().min(2),
  email: z.string().email(),
  password: z.string().min(8)
});

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { name, email, password } = signupSchema.parse(body);

    // Check if user exists
    const existingUser = await prisma.user.findUnique({
      where: { email }
    });

    if (existingUser) {
      return NextResponse.json(
        { error: 'User already exists' },
        { status: 400 }
      );
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 12);

    // Create user
    const user = await prisma.user.create({
      data: {
        name,
        email,
        password: hashedPassword
      }
    });

    // Create default workspace
    const workspace = await prisma.workspace.create({
      data: {
        name: `${name}'s Workspace`,
        slug: `${name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
        ownerId: user.id,
        plan: 'free',
        members: {
          create: {
            userId: user.id,
            role: 'owner'
          }
        },
        subscription: {
          create: {
            plan: 'free',
            status: 'active',
            currentPeriodEnd: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000) // 1 year
          }
        }
      }
    });

    return NextResponse.json({
      user: {
        id: user.id,
        name: user.name,
        email: user.email
      },
      workspace: {
        id: workspace.id,
        name: workspace.name
      }
    });
  } catch (error) {
    console.error('Signup error:', error);
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
