import { HttpClient } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';
import { Observable, catchError, map, of, shareReplay, tap, timer } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  AskRequest,
  AskResponse,
  GraphResponse,
  HealthResponse,
  MemoRequest,
  MemoResponse,
  NodeDetailResponse,
  RootResponse,
  StatsResponse,
} from '../models';

interface CacheEntry<T> {
  data: T;
  timestamp: number;
}

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private http = inject(HttpClient);
  private baseUrl = environment.apiBaseUrl;

  // Cache storage
  private graphCache = signal<CacheEntry<GraphResponse> | null>(null);
  private statsCache = signal<CacheEntry<StatsResponse> | null>(null);
  private nodeCache = new Map<string, CacheEntry<NodeDetailResponse>>();

  // Loading states
  readonly isLoadingGraph = signal(false);
  readonly isLoadingStats = signal(false);
  readonly isAskingQuestion = signal(false);
  readonly isAddingMemo = signal(false);

  // Error state
  readonly lastError = signal<string | null>(null);

  /**
   * Check if backend is healthy
   */
  checkHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.baseUrl}/health`);
  }

  /**
   * Get API info
   */
  getApiInfo(): Observable<RootResponse> {
    return this.http.get<RootResponse>(`${this.baseUrl}/`);
  }

  /**
   * Fetch the full knowledge graph (cached)
   */
  getGraph(forceRefresh = false): Observable<GraphResponse> {
    const cache = this.graphCache();
    const now = Date.now();

    // Return cached data if valid
    if (!forceRefresh && cache && now - cache.timestamp < environment.cacheConfig.graphTtl) {
      return of(cache.data);
    }

    this.isLoadingGraph.set(true);
    this.lastError.set(null);

    return this.http.get<GraphResponse>(`${this.baseUrl}/graph`).pipe(
      tap((data) => {
        this.graphCache.set({ data, timestamp: now });
        this.isLoadingGraph.set(false);
      }),
      catchError((error) => {
        this.isLoadingGraph.set(false);
        this.lastError.set(this.extractErrorMessage(error));
        throw error;
      })
    );
  }

  /**
   * Get graph statistics (cached)
   */
  getStats(forceRefresh = false): Observable<StatsResponse> {
    const cache = this.statsCache();
    const now = Date.now();

    if (!forceRefresh && cache && now - cache.timestamp < environment.cacheConfig.statsTtl) {
      return of(cache.data);
    }

    this.isLoadingStats.set(true);

    return this.http.get<StatsResponse>(`${this.baseUrl}/stats`).pipe(
      tap((data) => {
        this.statsCache.set({ data, timestamp: now });
        this.isLoadingStats.set(false);
      }),
      catchError((error) => {
        this.isLoadingStats.set(false);
        this.lastError.set(this.extractErrorMessage(error));
        throw error;
      })
    );
  }

  /**
   * Get detailed node information (cached)
   */
  getNode(nodeId: string, forceRefresh = false): Observable<NodeDetailResponse> {
    const cache = this.nodeCache.get(nodeId);
    const now = Date.now();

    if (!forceRefresh && cache && now - cache.timestamp < environment.cacheConfig.nodeTtl) {
      return of(cache.data);
    }

    return this.http.get<NodeDetailResponse>(`${this.baseUrl}/node/${nodeId}`).pipe(
      tap((data) => {
        this.nodeCache.set(nodeId, { data, timestamp: now });
      }),
      catchError((error) => {
        this.lastError.set(this.extractErrorMessage(error));
        throw error;
      })
    );
  }

  /**
   * Query the knowledge graph with RAG-enhanced response
   */
  ask(request: AskRequest): Observable<AskResponse> {
    this.isAskingQuestion.set(true);
    this.lastError.set(null);

    return this.http.post<AskResponse>(`${this.baseUrl}/ask`, request).pipe(
      tap(() => {
        this.isAskingQuestion.set(false);
        // Invalidate graph cache as new nodes may have been created
        this.invalidateGraphCache();
      }),
      catchError((error) => {
        this.isAskingQuestion.set(false);
        this.lastError.set(this.extractErrorMessage(error));
        throw error;
      })
    );
  }

  /**
   * Ingest a memo into the knowledge graph
   */
  addMemo(request: MemoRequest): Observable<MemoResponse> {
    this.isAddingMemo.set(true);
    this.lastError.set(null);

    return this.http.post<MemoResponse>(`${this.baseUrl}/memo`, request).pipe(
      tap(() => {
        this.isAddingMemo.set(false);
        // Invalidate caches as graph structure has changed
        this.invalidateGraphCache();
        this.invalidateStatsCache();
      }),
      catchError((error) => {
        this.isAddingMemo.set(false);
        this.lastError.set(this.extractErrorMessage(error));
        throw error;
      })
    );
  }

  /**
   * Invalidate graph cache (call after mutations)
   */
  invalidateGraphCache(): void {
    this.graphCache.set(null);
  }

  /**
   * Invalidate stats cache
   */
  invalidateStatsCache(): void {
    this.statsCache.set(null);
  }

  /**
   * Invalidate node cache for a specific node
   */
  invalidateNodeCache(nodeId?: string): void {
    if (nodeId) {
      this.nodeCache.delete(nodeId);
    } else {
      this.nodeCache.clear();
    }
  }

  /**
   * Clear all caches
   */
  clearAllCaches(): void {
    this.graphCache.set(null);
    this.statsCache.set(null);
    this.nodeCache.clear();
  }

  /**
   * Extract error message from HTTP error
   */
  private extractErrorMessage(error: unknown): string {
    if (error && typeof error === 'object') {
      const err = error as { error?: { detail?: string }; message?: string; status?: number };
      if (err.error?.detail) return err.error.detail;
      if (err.message) return err.message;
      if (err.status === 0) return 'Unable to connect to server';
      if (err.status === 404) return 'Resource not found';
      if (err.status === 500) return 'Server error';
    }
    return 'An unexpected error occurred';
  }
}
