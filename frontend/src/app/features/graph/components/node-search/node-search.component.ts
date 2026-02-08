import {
  Component,
  Input,
  Output,
  EventEmitter,
  signal,
  computed,
  ChangeDetectionStrategy,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { D3Node, NodeType } from '../../../../models';

@Component({
  selector: 'app-node-search',
  standalone: true,
  imports: [FormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="node-search glass-panel">
      <div class="search-input-wrapper">
        <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"/>
          <path d="M21 21l-4.35-4.35"/>
        </svg>
        <input
          type="text"
          class="search-input"
          placeholder="Search nodes..."
          [ngModel]="query()"
          (ngModelChange)="onQueryChange($event)"
          (focus)="showDropdown.set(true)"
          (blur)="onBlur()"
        />
        @if (query()) {
          <button class="clear-btn" (mousedown)="clearSearch($event)">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/>
            </svg>
          </button>
        }
      </div>
      @if (showDropdown() && filteredNodes().length > 0) {
        <div class="search-results">
          @for (node of filteredNodes().slice(0, 10); track node.id) {
            <button
              class="search-result"
              (mousedown)="selectNode(node, $event)"
            >
              <span class="node-type-badge" [class]="'node-type-badge--' + node.metadata.type">
                {{ getTypeLabel(node.metadata.type) }}
              </span>
              <span class="node-document">{{ truncate(node.document, 50) }}</span>
            </button>
          }
          @if (filteredNodes().length > 10) {
            <div class="more-results">
              +{{ filteredNodes().length - 10 }} more results
            </div>
          }
        </div>
      }
    </div>
  `,
  styles: [`
    .node-search {
      width: 280px;
      overflow: hidden;
    }

    .search-input-wrapper {
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
      padding: var(--spacing-sm) var(--spacing-md);
    }

    .search-icon {
      width: 18px;
      height: 18px;
      color: var(--color-text-muted);
      flex-shrink: 0;
    }

    .search-input {
      flex: 1;
      background: none;
      border: none;
      padding: 0;
      color: var(--color-text-primary);
      font-size: 14px;

      &:focus {
        outline: none;
        box-shadow: none;
      }

      &::placeholder {
        color: var(--color-text-muted);
      }
    }

    .clear-btn {
      width: 18px;
      height: 18px;
      padding: 0;
      background: none;
      border: none;
      color: var(--color-text-muted);
      cursor: pointer;
      flex-shrink: 0;

      &:hover {
        color: var(--color-text-primary);
      }

      svg {
        width: 100%;
        height: 100%;
      }
    }

    .search-results {
      border-top: 1px solid var(--glass-border);
      max-height: 300px;
      overflow-y: auto;
    }

    .search-result {
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
      width: 100%;
      padding: var(--spacing-sm) var(--spacing-md);
      background: none;
      border: none;
      text-align: left;
      cursor: pointer;
      transition: background var(--transition-fast);

      &:hover {
        background: var(--color-bg-tertiary);
      }
    }

    .node-type-badge {
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      padding: 2px 6px;
      border-radius: var(--radius-sm);

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

    .node-document {
      flex: 1;
      font-size: 13px;
      color: var(--color-text-primary);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .more-results {
      padding: var(--spacing-sm) var(--spacing-md);
      font-size: 12px;
      color: var(--color-text-muted);
      text-align: center;
    }
  `],
})
export class NodeSearchComponent {
  @Input() set nodes(value: D3Node[]) {
    this._nodes.set(value);
  }

  @Output() nodeSelect = new EventEmitter<D3Node>();
  @Output() queryChange = new EventEmitter<string>();

  private _nodes = signal<D3Node[]>([]);
  readonly query = signal('');
  readonly showDropdown = signal(false);

  readonly filteredNodes = computed(() => {
    const q = this.query().toLowerCase();
    if (!q) return [];
    return this._nodes().filter(
      (n) =>
        n.document.toLowerCase().includes(q) ||
        n.id.toLowerCase().includes(q)
    );
  });

  onQueryChange(value: string): void {
    this.query.set(value);
    this.queryChange.emit(value);
  }

  selectNode(node: D3Node, event: MouseEvent): void {
    event.preventDefault();
    this.nodeSelect.emit(node);
    this.query.set('');
    this.showDropdown.set(false);
  }

  clearSearch(event: MouseEvent): void {
    event.preventDefault();
    this.query.set('');
    this.queryChange.emit('');
  }

  onBlur(): void {
    // Delay to allow click on results
    setTimeout(() => this.showDropdown.set(false), 150);
  }

  getTypeLabel(type: NodeType): string {
    return {
      [NodeType.CONCEPT]: 'C',
      [NodeType.ENTITY]: 'E',
      [NodeType.TAG]: 'T',
    }[type] || '?';
  }

  truncate(text: string, maxLength: number): string {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  }
}
