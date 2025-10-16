import { NextRequest, NextResponse } from 'next/server';
import Stripe from 'stripe';
import prisma from '@/lib/prisma';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY || '', {
  apiVersion: '2024-12-18.acacia'
});

const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET || '';

export async function POST(req: NextRequest) {
  try {
    const body = await req.text();
    const signature = req.headers.get('stripe-signature');

    if (!signature) {
      return NextResponse.json({ error: 'No signature' }, { status: 400 });
    }

    let event: Stripe.Event;

    try {
      event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
    } catch (err) {
      console.error('Webhook signature verification failed:', err);
      return NextResponse.json({ error: 'Invalid signature' }, { status: 400 });
    }

    // Handle the event
    switch (event.type) {
      case 'checkout.session.completed': {
        const session = event.data.object as Stripe.Checkout.Session;
        const workspaceId = session.metadata?.workspaceId;
        const plan = session.metadata?.plan;

        if (workspaceId && plan) {
          await prisma.subscription.update({
            where: { workspaceId },
            data: {
              plan,
              status: 'active',
              stripeSubscriptionId: session.subscription as string,
              currentPeriodStart: new Date(),
              currentPeriodEnd: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000)
            }
          });

          await prisma.workspace.update({
            where: { id: workspaceId },
            data: { plan }
          });
        }
        break;
      }

      case 'customer.subscription.updated': {
        const subscription = event.data.object as Stripe.Subscription;
        const workspaceSubscription = await prisma.subscription.findFirst({
          where: { stripeSubscriptionId: subscription.id }
        });

        if (workspaceSubscription) {
          await prisma.subscription.update({
            where: { id: workspaceSubscription.id },
            data: {
              status: subscription.status,
              currentPeriodEnd: new Date(subscription.current_period_end * 1000)
            }
          });
        }
        break;
      }

      case 'customer.subscription.deleted': {
        const subscription = event.data.object as Stripe.Subscription;
        const workspaceSubscription = await prisma.subscription.findFirst({
          where: { stripeSubscriptionId: subscription.id }
        });

        if (workspaceSubscription) {
          await prisma.subscription.update({
            where: { id: workspaceSubscription.id },
            data: {
              status: 'canceled',
              plan: 'free'
            }
          });

          await prisma.workspace.update({
            where: { id: workspaceSubscription.workspaceId },
            data: { plan: 'free' }
          });
        }
        break;
      }

      case 'invoice.payment_failed': {
        const invoice = event.data.object as Stripe.Invoice;
        const workspaceSubscription = await prisma.subscription.findFirst({
          where: { stripeCustomerId: invoice.customer as string }
        });

        if (workspaceSubscription) {
          await prisma.subscription.update({
            where: { id: workspaceSubscription.id },
            data: { status: 'past_due' }
          });
        }
        break;
      }
    }

    return NextResponse.json({ received: true });
  } catch (error) {
    console.error('Webhook error:', error);
    return NextResponse.json(
      { error: 'Webhook handler failed' },
      { status: 500 }
    );
  }
}
