import { Injectable, signal, computed, inject } from '@angular/core';
import { D3Node, D3Link, GraphNode, Edge, NodeType } from '../../models';
import { ApiService } from '../../core';

@Injectable({
  providedIn: 'root',
})
export class GraphService {
  private apiService = inject(ApiService);

  // Graph data
  readonly nodes = signal<D3Node[]>([]);
  readonly links = signal<D3Link[]>([]);

  // Selection state
  readonly selectedNodeId = signal<string | null>(null);
  readonly hoveredNodeId = signal<string | null>(null);

  // Search state
  readonly searchQuery = signal('');

  // Computed
  readonly selectedNode = computed(() => {
    const id = this.selectedNodeId();
    return id ? this.nodes().find((n) => n.id === id) : null;
  });

  readonly filteredNodes = computed(() => {
    const query = this.searchQuery().toLowerCase();
    if (!query) return this.nodes();
    return this.nodes().filter(
      (n) =>
        n.document.toLowerCase().includes(query) ||
        n.id.toLowerCase().includes(query)
    );
  });

  readonly nodeCount = computed(() => this.nodes().length);
  readonly edgeCount = computed(() => this.links().length);

  /**
   * Load graph data from API
   */
  async loadGraph(forceRefresh = false): Promise<void> {
    try {
      const response = await this.apiService.getGraph(forceRefresh).toPromise();
      if (response) {
        const nodes: D3Node[] = response.nodes.map((n) => ({
          ...n,
          metadata: {
            ...n.metadata,
            type: n.metadata.type as NodeType,
          },
        }));

        const links: D3Link[] = response.edges.map((e) => ({
          source: e.source,
          target: e.target,
          weight: e.weight,
          relation_type: e.relation_type,
        }));

        this.nodes.set(nodes);
        this.links.set(links);
      }
    } catch (error) {
      console.error('Failed to load graph:', error);
    }
  }

  /**
   * Select a node
   */
  selectNode(nodeId: string | null): void {
    this.selectedNodeId.set(nodeId);
  }

  /**
   * Set hovered node
   */
  hoverNode(nodeId: string | null): void {
    this.hoveredNodeId.set(nodeId);
  }

  /**
   * Get connected node IDs for a given node
   */
  getConnectedNodeIds(nodeId: string): Set<string> {
    const connected = new Set<string>();
    this.links().forEach((link) => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target.id;
      if (sourceId === nodeId) connected.add(targetId);
      if (targetId === nodeId) connected.add(sourceId);
    });
    return connected;
  }

  /**
   * Get node size based on type and importance
   */
  getNodeSize(node: D3Node): number {
    const baseSize = {
      [NodeType.CONCEPT]: 24,
      [NodeType.ENTITY]: 18,
      [NodeType.TAG]: 12,
    }[node.metadata.type] || 16;

    // Scale by importance (1.0 to 1.5x)
    const importanceScale = 1 + node.metadata.importance_score * 0.5;
    return baseSize * importanceScale;
  }

  /**
   * Get node color based on type
   */
  getNodeColor(node: D3Node): string {
    return {
      [NodeType.CONCEPT]: 'var(--color-node-concept)',
      [NodeType.ENTITY]: 'var(--color-node-entity)',
      [NodeType.TAG]: 'var(--color-node-tag)',
    }[node.metadata.type] || 'var(--color-accent-primary)';
  }

  /**
   * Update search query
   */
  setSearchQuery(query: string): void {
    this.searchQuery.set(query);
  }

  /**
   * Clear selection
   */
  clearSelection(): void {
    this.selectedNodeId.set(null);
    this.hoveredNodeId.set(null);
  }
}
