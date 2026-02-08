/**
 * TypeScript interfaces matching backend Pydantic models
 */

export enum NodeType {
  CONCEPT = 'concept',
  TAG = 'tag',
  ENTITY = 'entity',
}

export interface NodeMetadata {
  type: NodeType;
  created_at: number;
  last_interaction: number;
  connection_count: number;
  importance_score: number;
  original_source_id?: string | null;
}

export interface GraphNode {
  id: string;
  document: string;
  metadata: NodeMetadata;
  embedding?: number[] | null;
}

export interface Edge {
  source: string;
  target: string;
  weight: number;
  relation_type: string;
}

/**
 * D3-specific types for force simulation
 */
export interface D3Node extends GraphNode {
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface D3Link {
  source: D3Node | string;
  target: D3Node | string;
  weight: number;
  relation_type: string;
}

/**
 * Graph data structure for visualization
 */
export interface GraphData {
  nodes: GraphNode[];
  edges: Edge[];
}

/**
 * Graph statistics from backend
 */
export interface GraphStats {
  total_nodes: number;
  total_edges: number;
  node_types: {
    concept: number;
    tag: number;
    entity: number;
  };
  avg_connections: number;
}
