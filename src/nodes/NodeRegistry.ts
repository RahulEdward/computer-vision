import { INodeType, INodeTypeDescription, NODE_CATEGORIES } from '../types/n8nTypes';
import { BaseNode } from './base/BaseNode';
import { ManualTriggerNode, WebhookTriggerNode, ScheduleTriggerNode } from './trigger/TriggerNode';
import { HttpRequestNode, DataTransformNode, ConditionNode } from './action/ActionNode';
import { TimerNode } from './TimerNode';
import { HttpNode } from './HttpNode';
import { TransformNode } from './TransformNode';

export interface NodeCategory {
  name: string;
  displayName: string;
  description: string;
  icon: string;
  color: string;
}

export class NodeRegistry {
  private static instance: NodeRegistry;
  private nodes: Map<string, () => INodeType> = new Map();
  private categories: Map<string, NodeCategory> = new Map();
  private nodesByCategory: Map<string, string[]> = new Map();

  private constructor() {
    this.initializeCategories();
    this.registerDefaultNodes();
  }

  public static getInstance(): NodeRegistry {
    if (!NodeRegistry.instance) {
      NodeRegistry.instance = new NodeRegistry();
    }
    return NodeRegistry.instance;
  }

  private initializeCategories(): void {
    const categories: NodeCategory[] = [
      {
        name: NODE_CATEGORIES.TRIGGER,
        displayName: 'Triggers',
        description: 'Start workflows with various triggers',
        icon: 'fa:play',
        color: 'from-green-500 to-green-600'
      },
      {
        name: NODE_CATEGORIES.ACTION,
        displayName: 'Actions',
        description: 'Perform actions and operations',
        icon: 'fa:cog',
        color: 'from-blue-500 to-blue-600'
      },
      {
        name: NODE_CATEGORIES.TRANSFORM,
        displayName: 'Transform',
        description: 'Transform and manipulate data',
        icon: 'fa:exchange-alt',
        color: 'from-purple-500 to-purple-600'
      },
      {
        name: NODE_CATEGORIES.CORE,
        displayName: 'Core',
        description: 'Core workflow functionality',
        icon: 'fa:cube',
        color: 'from-gray-500 to-gray-600'
      },
      {
        name: NODE_CATEGORIES.INTEGRATION,
        displayName: 'Integrations',
        description: 'Connect with external services',
        icon: 'fa:plug',
        color: 'from-indigo-500 to-indigo-600'
      },
      {
        name: NODE_CATEGORIES.UTILITY,
        displayName: 'Utilities',
        description: 'Utility and helper functions',
        icon: 'fa:tools',
        color: 'from-yellow-500 to-yellow-600'
      }
    ];

    for (const category of categories) {
      this.categories.set(category.name, category);
      this.nodesByCategory.set(category.name, []);
    }
  }

  private registerDefaultNodes(): void {
    // Register new modular nodes
    this.registerNode('manualTrigger', () => new ManualTriggerNode());
    this.registerNode('webhook', () => new WebhookTriggerNode());
    this.registerNode('scheduleTrigger', () => new ScheduleTriggerNode());
    this.registerNode('httpRequest', () => new HttpRequestNode());
    this.registerNode('dataTransform', () => new DataTransformNode());
    this.registerNode('condition', () => new ConditionNode());
    
    // Keep existing nodes for backward compatibility
    this.registerNode('timer', () => new TimerNode());
    this.registerNode('httpRequestLegacy', () => new HttpNode());
    this.registerNode('transform', () => new TransformNode());
  }

  public registerNode(name: string, nodeFactory: () => INodeType): void {
    this.nodes.set(name, nodeFactory);
    
    // Add to category mapping
    const node = nodeFactory();
    const categories = node.description.group;
    for (const category of categories) {
      if (!this.nodesByCategory.has(category)) {
        this.nodesByCategory.set(category, []);
      }
      this.nodesByCategory.get(category)!.push(name);
    }
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
    return Array.from(this.categories.keys());
  }

  public getCategoryInfo(categoryName: string): NodeCategory | null {
    return this.categories.get(categoryName) || null;
  }

  public getAllCategories(): NodeCategory[] {
    return Array.from(this.categories.values());
  }

  public getNodeNamesInCategory(category: string): string[] {
    return this.nodesByCategory.get(category) || [];
  }

  public searchNodes(query: string): INodeTypeDescription[] {
    const lowerQuery = query.toLowerCase();
    return this.getAllNodeDescriptions().filter(desc => 
      desc.displayName.toLowerCase().includes(lowerQuery) ||
      desc.description.toLowerCase().includes(lowerQuery) ||
      desc.name.toLowerCase().includes(lowerQuery)
    );
  }

  public getNodeCount(): number {
    return this.nodes.size;
  }

  public isNodeRegistered(name: string): boolean {
    return this.nodes.has(name);
  }
}

// Export singleton instance
export const nodeRegistry = NodeRegistry.getInstance();