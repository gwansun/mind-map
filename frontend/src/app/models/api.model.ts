/**
 * API request/response types matching backend FastAPI routes
 */

import { Edge, GraphNode, NodeMetadata } from './graph.model';

// GET / response
export interface RootResponse {
  message: string;
  version: string;
}

// GET /health response
export interface HealthResponse {
  status: string;
}

// GET /graph response
export interface GraphResponse {
  nodes: Array<{
    id: string;
    document: string;
    metadata: NodeMetadata;
  }>;
  edges: Array<{
    source: string;
    target: string;
    weight: number;
    relation_type: string;
  }>;
}

// GET /node/{id} response
export interface NodeDetailResponse {
  node: GraphNode;
  edges: Edge[];
  importance_score: number;
}

// POST /ask request
export interface AskRequest {
  query: string;
  depth?: number;
}

// POST /ask response
export interface AskResponse {
  query: string;
  response: string;
  context_nodes: string[];
}

// POST /memo request
export interface MemoRequest {
  text: string;
  source?: string | null;
}

// POST /memo response
export interface MemoResponse {
  status: 'success' | 'skipped';
  message: string;
  node_ids: string[];
}

// GET /stats response
export interface StatsResponse {
  total_nodes: number;
  total_edges: number;
  node_types: {
    concept: number;
    tag: number;
    entity: number;
  };
  avg_connections: number;
}
