import {
  Component,
  Input,
  Output,
  EventEmitter,
  signal,
  effect,
  inject,
  ChangeDetectionStrategy,
} from '@angular/core';
import { DatePipe } from '@angular/common';
import { ApiService } from '../../core';
import { D3Node, Edge, NodeType, NodeDetailResponse } from '../../models';
import { LoadingSpinnerComponent } from '../../shared';

@Component({
  selector: 'app-inspector-panel',
  standalone: true,
  imports: [DatePipe, LoadingSpinnerComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="inspector-panel glass-panel">
      <div class="inspector-header">
        <h3>Node Inspector</h3>
        @if (node) {
          <button class="btn btn--ghost btn--icon" (click)="close.emit()" title="Close">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        }
      </div>

      @if (!node) {
        <div class="inspector-empty">
          <p>Select a node to view details</p>
        </div>
      } @else {
        @if (isLoading()) {
          <app-loading-spinner message="Loading details..."></app-loading-spinner>
        } @else {
          <div class="inspector-content">
            <div class="node-type-badge" [class]="'node-type-badge--' + node.metadata.type">
              {{ getTypeLabel(node.metadata.type) }}
            </div>

            <section class="inspector-section">
              <h4>Content</h4>
              <p class="node-document">{{ node.document }}</p>
            </section>

            <section class="inspector-section">
              <h4>Metadata</h4>
              <dl class="metadata-list">
                <div class="metadata-item">
                  <dt>ID</dt>
                  <dd class="mono">{{ node.id.substring(0, 16) }}...</dd>
                </div>
                <div class="metadata-item">
                  <dt>Created</dt>
                  <dd>{{ node.metadata.created_at * 1000 | date:'medium' }}</dd>
                </div>
                <div class="metadata-item">
                  <dt>Last Interaction</dt>
                  <dd>{{ node.metadata.last_interaction * 1000 | date:'medium' }}</dd>
                </div>
                <div class="metadata-item">
                  <dt>Connections</dt>
                  <dd>{{ node.metadata.connection_count }}</dd>
                </div>
                <div class="metadata-item">
                  <dt>Importance</dt>
                  <dd>
                    <div class="importance-bar">
                      <div class="importance-fill" [style.width.%]="importanceScore() * 100"></div>
                    </div>
                    <span class="importance-value">{{ (importanceScore() * 100).toFixed(1) }}%</span>
                  </dd>
                </div>
              </dl>
            </section>

            @if (edges().length > 0) {
              <section class="inspector-section">
                <h4>Connections ({{ edges().length }})</h4>
                <ul class="edge-list">
                  @for (edge of edges().slice(0, 10); track edge.source + edge.target) {
                    <li class="edge-item">
                      <span class="edge-relation">{{ edge.relation_type }}</span>
                      <button
                        class="edge-target"
                        (click)="selectNode.emit(edge.source === node.id ? edge.target : edge.source)"
                      >
                        {{ (edge.source === node.id ? edge.target : edge.source).substring(0, 8) }}...
                      </button>
                    </li>
                  }
                  @if (edges().length > 10) {
                    <li class="edge-more">+{{ edges().length - 10 }} more connections</li>
                  }
                </ul>
              </section>
            }
          </div>
        }
      }
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100%;
    }

    .inspector-panel {
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
    }

    .inspector-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: var(--spacing-md);
      border-bottom: 1px solid var(--glass-border);

      h3 {
        font-size: 14px;
        font-weight: 600;
        color: var(--color-text-primary);
      }
    }

    .inspector-empty {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: var(--spacing-lg);
      color: var(--color-text-muted);
      text-align: center;
    }

    .inspector-content {
      flex: 1;
      overflow-y: auto;
      padding: var(--spacing-md);
    }

    .node-type-badge {
      display: inline-block;
      padding: var(--spacing-xs) var(--spacing-sm);
      border-radius: var(--radius-sm);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      margin-bottom: var(--spacing-md);

      &--concept {
        background: rgba(99, 102, 241, 0.2);
        color: var(--color-node-concept);
      }

      &--entity {
        background: rgba(34, 197, 94, 0.2);
        color: var(--color-node-entity);
      }

      &--tag {
        background: rgba(245, 158, 11, 0.2);
        color: var(--color-node-tag);
      }
    }

    .inspector-section {
      margin-bottom: var(--spacing-lg);

      h4 {
        font-size: 12px;
        font-weight: 600;
        color: var(--color-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: var(--spacing-sm);
      }
    }

    .node-document {
      font-size: 14px;
      line-height: 1.6;
      color: var(--color-text-primary);
    }

    .metadata-list {
      display: flex;
      flex-direction: column;
      gap: var(--spacing-sm);
    }

    .metadata-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: var(--spacing-sm);

      dt {
        font-size: 12px;
        color: var(--color-text-muted);
      }

      dd {
        font-size: 12px;
        color: var(--color-text-primary);
        text-align: right;

        &.mono {
          font-family: var(--font-mono);
          font-size: 11px;
        }
      }
    }

    .importance-bar {
      display: inline-block;
      width: 60px;
      height: 6px;
      background: var(--color-bg-tertiary);
      border-radius: var(--radius-full);
      overflow: hidden;
      margin-right: var(--spacing-xs);
    }

    .importance-fill {
      height: 100%;
      background: var(--color-accent-primary);
      border-radius: var(--radius-full);
      transition: width var(--transition-normal);
    }

    .importance-value {
      font-family: var(--font-mono);
      font-size: 11px;
    }

    .edge-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: var(--spacing-xs);
    }

    .edge-item {
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
      padding: var(--spacing-xs) var(--spacing-sm);
      background: var(--color-bg-tertiary);
      border-radius: var(--radius-sm);
    }

    .edge-relation {
      font-size: 11px;
      color: var(--color-text-muted);
      flex-shrink: 0;
    }

    .edge-target {
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 2px 6px;
      background: var(--color-bg-elevated);
      border: 1px solid var(--glass-border);
      border-radius: var(--radius-sm);
      color: var(--color-accent-secondary);
      cursor: pointer;
      transition: all var(--transition-fast);

      &:hover {
        background: var(--color-accent-primary);
        color: white;
      }
    }

    .edge-more {
      font-size: 11px;
      color: var(--color-text-muted);
      text-align: center;
      padding: var(--spacing-xs);
    }
  `],
})
export class InspectorPanelComponent {
  private apiService = inject(ApiService);

  @Input() set node(value: D3Node | null) {
    this._node = value;
    if (value) {
      this.loadNodeDetails(value.id);
    } else {
      this.edges.set([]);
      this.importanceScore.set(0);
    }
  }
  get node(): D3Node | null {
    return this._node;
  }

  @Output() close = new EventEmitter<void>();
  @Output() selectNode = new EventEmitter<string>();

  private _node: D3Node | null = null;

  readonly isLoading = signal(false);
  readonly edges = signal<Edge[]>([]);
  readonly importanceScore = signal(0);

  private async loadNodeDetails(nodeId: string): Promise<void> {
    this.isLoading.set(true);
    try {
      const details = await this.apiService.getNode(nodeId).toPromise();
      if (details) {
        this.edges.set(details.edges);
        this.importanceScore.set(details.importance_score);
      }
    } catch (error) {
      console.error('Failed to load node details:', error);
    } finally {
      this.isLoading.set(false);
    }
  }

  getTypeLabel(type: NodeType): string {
    return {
      [NodeType.CONCEPT]: 'Concept',
      [NodeType.ENTITY]: 'Entity',
      [NodeType.TAG]: 'Tag',
    }[type] || 'Unknown';
  }
}
