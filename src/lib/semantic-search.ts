import Fuse from 'fuse.js';
import * as tf from '@tensorflow/tfjs';

interface SearchableItem {
  id: string;
  title: string;
  description: string;
  tags: string[];
  type: 'workflow' | 'node' | 'template' | 'documentation';
  content?: string;
  metadata?: Record<string, any>;
}

export class SemanticSearchEngine {
  private fuse: Fuse<SearchableItem>;
  private items: SearchableItem[] = [];
  private embeddings: Map<string, number[]> = new Map();
  private model: tf.LayersModel | null = null;

  constructor() {
    this.fuse = new Fuse([], {
      keys: [
        { name: 'title', weight: 0.3 },
        { name: 'description', weight: 0.2 },
        { name: 'tags', weight: 0.2 },
        { name: 'content', weight: 0.3 }
      ],
      threshold: 0.4,
      includeScore: true,
      includeMatches: true
    });

    this.initializeModel();
  }

  private async initializeModel() {
    try {
      // Semantic model is optional - using fuzzy search by default
      // To enable semantic search, add a TensorFlow model to /public/models/
      // this.model = await tf.loadLayersModel('/models/sentence-encoder/model.json');
      console.log('Using fuzzy search (semantic model disabled)');
    } catch (error) {
      console.warn('Semantic model not available, using fuzzy search');
    }
  }

  addItems(items: SearchableItem[]) {
    this.items = [...this.items, ...items];
    this.fuse.setCollection(this.items);
    
    // Generate embeddings for semantic search
    items.forEach(item => {
      this.generateEmbedding(item);
    });
  }

  private async generateEmbedding(item: SearchableItem) {
    if (!this.model) return;

    try {
      const text = `${item.title} ${item.description} ${item.tags.join(' ')} ${item.content || ''}`;
      const tokens = this.tokenize(text);
      const tensor = tf.tensor2d([tokens]);
      const embedding = await this.model.predict(tensor) as tf.Tensor;
      const embeddingArray = await embedding.data();
      
      this.embeddings.set(item.id, Array.from(embeddingArray));
      
      tensor.dispose();
      embedding.dispose();
    } catch (error) {
      console.error('Error generating embedding:', error);
    }
  }

  private tokenize(text: string): number[] {
    // Simple tokenization - in production use proper tokenizer
    const words = text.toLowerCase().split(/\W+/).filter(w => w.length > 0);
    const vocab = this.getVocabulary();
    
    return words.slice(0, 128).map(word => vocab.get(word) || 0);
  }

  private getVocabulary(): Map<string, number> {
    // Simple vocabulary - in production use proper vocabulary
    const commonWords = [
      'workflow', 'automation', 'trigger', 'action', 'data', 'api', 'http',
      'email', 'file', 'database', 'schedule', 'condition', 'loop', 'transform',
      'webhook', 'notification', 'integration', 'process', 'execute', 'run'
    ];
    
    const vocab = new Map<string, number>();
    commonWords.forEach((word, index) => {
      vocab.set(word, index + 1);
    });
    
    return vocab;
  }

  async search(query: string, options: {
    type?: string[];
    limit?: number;
    useSemanticSearch?: boolean;
  } = {}): Promise<{
    items: SearchableItem[];
    scores: number[];
    matches: any[];
  }> {
    const { type, limit = 10, useSemanticSearch = true } = options;

    // Fuzzy search results
    let fuzzyResults = this.fuse.search(query, { limit });
    
    if (type && type.length > 0) {
      fuzzyResults = fuzzyResults.filter(result => 
        type.includes(result.item.type)
      );
    }

    // Semantic search results (if model is available)
    let semanticResults: any[] = [];
    if (useSemanticSearch && this.model && this.embeddings.size > 0) {
      semanticResults = await this.performSemanticSearch(query, type);
    }

    // Combine and rank results
    const combinedResults = this.combineResults(fuzzyResults, semanticResults);
    
    return {
      items: combinedResults.slice(0, limit).map(r => r.item),
      scores: combinedResults.slice(0, limit).map(r => r.score),
      matches: combinedResults.slice(0, limit).map(r => r.matches || [])
    };
  }

  private async performSemanticSearch(query: string, type?: string[]): Promise<any[]> {
    if (!this.model) return [];

    try {
      const queryTokens = this.tokenize(query);
      const queryTensor = tf.tensor2d([queryTokens]);
      const queryEmbedding = await this.model.predict(queryTensor) as tf.Tensor;
      const queryVector = await queryEmbedding.data();

      const similarities: { item: SearchableItem; score: number }[] = [];

      for (const [itemId, embedding] of this.embeddings) {
        const item = this.items.find(i => i.id === itemId);
        if (!item) continue;
        
        if (type && type.length > 0 && !type.includes(item.type)) {
          continue;
        }

        const similarity = this.cosineSimilarity(Array.from(queryVector), embedding);
        similarities.push({ item, score: similarity });
      }

      queryTensor.dispose();
      queryEmbedding.dispose();

      return similarities
        .sort((a, b) => b.score - a.score)
        .map(s => ({ item: s.item, score: s.score }));

    } catch (error) {
      console.error('Semantic search error:', error);
      return [];
    }
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    
    return dotProduct / (magnitudeA * magnitudeB);
  }

  private combineResults(fuzzyResults: any[], semanticResults: any[]): any[] {
    const combined = new Map<string, any>();

    // Add fuzzy results with weight
    fuzzyResults.forEach(result => {
      const score = 1 - (result.score || 0); // Fuse.js uses lower scores for better matches
      combined.set(result.item.id, {
        item: result.item,
        score: score * 0.6, // Weight fuzzy search at 60%
        matches: result.matches
      });
    });

    // Add semantic results with weight
    semanticResults.forEach(result => {
      const existing = combined.get(result.item.id);
      if (existing) {
        // Combine scores
        existing.score = existing.score + (result.score * 0.4); // Weight semantic at 40%
      } else {
        combined.set(result.item.id, {
          item: result.item,
          score: result.score * 0.4,
          matches: []
        });
      }
    });

    return Array.from(combined.values()).sort((a, b) => b.score - a.score);
  }

  // Auto-complete suggestions
  getSuggestions(query: string, limit = 5): string[] {
    const results = this.fuse.search(query, { limit });
    return results.map(r => r.item.title);
  }

  // Get related items
  getRelated(itemId: string, limit = 5): SearchableItem[] {
    const item = this.items.find(i => i.id === itemId);
    if (!item) return [];

    const relatedQuery = `${item.tags.join(' ')} ${item.type}`;
    const results = this.fuse.search(relatedQuery, { limit: limit + 1 });
    
    return results
      .filter(r => r.item.id !== itemId)
      .slice(0, limit)
      .map(r => r.item);
  }

  // Clear all data
  clear() {
    this.items = [];
    this.embeddings.clear();
    this.fuse.setCollection([]);
  }
}

// React hook for semantic search
import { useEffect, useRef, useState } from 'react';

export function useSemanticSearch() {
  const searchEngine = useRef<SemanticSearchEngine | null>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    searchEngine.current = new SemanticSearchEngine();
    setIsReady(true);
  }, []);

  const search = async (query: string, options?: any) => {
    if (!searchEngine.current) return { items: [], scores: [], matches: [] };
    return searchEngine.current.search(query, options);
  };

  const addItems = (items: SearchableItem[]) => {
    searchEngine.current?.addItems(items);
  };

  const getSuggestions = (query: string, limit?: number) => {
    return searchEngine.current?.getSuggestions(query, limit) || [];
  };

  const getRelated = (itemId: string, limit?: number) => {
    return searchEngine.current?.getRelated(itemId, limit) || [];
  };

  return {
    search,
    addItems,
    getSuggestions,
    getRelated,
    isReady
  };
}