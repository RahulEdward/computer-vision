import { Workspace, WorkspaceMember, User } from '@/types/saas';

class WorkspaceService {
  private workspaces: Map<string, Workspace> = new Map();

  async createWorkspace(name: string, ownerId: string): Promise<Workspace> {
    const workspace: Workspace = {
      id: this.generateId(),
      name,
      slug: this.slugify(name),
      ownerId,
      plan: 'free',
      members: [
        {
          userId: ownerId,
          role: 'owner',
          joinedAt: new Date()
        }
      ],
      settings: {
        allowInvites: true,
        maxMembers: 1
      },
      createdAt: new Date(),
      updatedAt: new Date()
    };

    this.workspaces.set(workspace.id, workspace);
    return workspace;
  }

  async getWorkspace(id: string): Promise<Workspace | null> {
    return this.workspaces.get(id) || null;
  }

  async getUserWorkspaces(userId: string): Promise<Workspace[]> {
    return Array.from(this.workspaces.values()).filter(ws =>
      ws.members.some(m => m.userId === userId)
    );
  }

  async addMember(
    workspaceId: string,
    userId: string,
    role: 'admin' | 'member' = 'member'
  ): Promise<boolean> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) return false;

    if (workspace.members.some(m => m.userId === userId)) {
      return false; // Already a member
    }

    workspace.members.push({
      userId,
      role,
      joinedAt: new Date()
    });

    workspace.updatedAt = new Date();
    return true;
  }

  async removeMember(workspaceId: string, userId: string): Promise<boolean> {
    const workspace = this.workspaces.get(workspaceId);
    if (!workspace) return false;

    const memberIndex = workspace.members.findIndex(m => m.userId === userId);
    if (memberIndex === -1) return false;

    // Can't remove owner
    if (workspace.members[memberIndex].role === 'owner') return false;

    workspace.members.splice(memberIndex, 1);
    workspace.updatedAt = new Date();
    return true;
  }

  async updateWorkspace(
    id: string,
    updates: Partial<Workspace>
  ): Promise<Workspace | null> {
    const workspace = this.workspaces.get(id);
    if (!workspace) return null;

    Object.assign(workspace, updates, { updatedAt: new Date() });
    return workspace;
  }

  private generateId(): string {
    return `ws_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private slugify(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/[\s_-]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }
}

export const workspaceService = new WorkspaceService();
