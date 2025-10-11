import { INodeType, INodeTypeDescription, NODE_CATEGORIES } from '../types/n8nTypes';
import { TimerNode } from './TimerNode';
import { HttpNode } from './HttpNode';
import { TransformNode } from './TransformNode';

export class NodeRegistry {
  private static instance: NodeRegistry;
  private nodes: Map<string, () => INodeType> = new Map();

  private constructor() {
    this.registerDefaultNodes();
  }

  public static getInstance(): NodeRegistry {
    if (!NodeRegistry.instance) {
      NodeRegistry.instance = new NodeRegistry();
    }
    return NodeRegistry.instance;
  }

  private registerDefaultNodes(): void {
    // Register core nodes
    this.registerNode('timer', () => new TimerNode());
    this.registerNode('httpRequest', () => new HttpNode());
    this.registerNode('transform', () => new TransformNode());
  }

  public registerNode(name: string, nodeFactory: () => INodeType): void {
    this.nodes.set(name, nodeFactory);
  }

  public getNode(name: string): INodeType | null {
    const factory = this.nodes.get(name);
    return factory ? factory() : null;
  }

  public getAllNodeDescriptions(): INodeTypeDescription[] {
    const descriptions: INodeTypeDescription[] = [];
    
    for (const factory of this.nodes.values()) {
      const node = factory();
      descriptions.push(node.description);
    }
    
    return descriptions;
  }

  public getNodesByCategory(category: string): INodeTypeDescription[] {
    return this.getAllNodeDescriptions().filter(desc => 
      desc.group.includes(category)
    );
  }

  public getAvailableCategories(): string[] {
    const categories = new Set<string>();
    
    this.getAllNodeDescriptions().forEach(desc => {
      desc.group.forEach(group => categories.add(group));
    });
    
    return Array.from(categories);
  }

  public searchNodes(query: string): INodeTypeDescription[] {
    const lowerQuery = query.toLowerCase();
    
    return this.getAllNodeDescriptions().filter(desc =>
      desc.displayName.toLowerCase().includes(lowerQuery) ||
      desc.description.toLowerCase().includes(lowerQuery) ||
      desc.name.toLowerCase().includes(lowerQuery)
    );
  }
}

// Export singleton instance
export const nodeRegistry = NodeRegistry.getInstance();