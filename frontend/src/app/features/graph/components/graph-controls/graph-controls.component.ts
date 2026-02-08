import { Component, EventEmitter, Output, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-graph-controls',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="graph-controls glass-panel">
      <button class="btn btn--ghost btn--icon" (click)="zoomIn.emit()" title="Zoom In">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"/>
          <path d="M21 21l-4.35-4.35M11 8v6M8 11h6"/>
        </svg>
      </button>
      <button class="btn btn--ghost btn--icon" (click)="zoomOut.emit()" title="Zoom Out">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"/>
          <path d="M21 21l-4.35-4.35M8 11h6"/>
        </svg>
      </button>
      <div class="divider"></div>
      <button class="btn btn--ghost btn--icon" (click)="resetView.emit()" title="Reset View">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
          <path d="M21 3v5h-5"/>
          <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
          <path d="M3 21v-5h5"/>
        </svg>
      </button>
      <button class="btn btn--ghost btn--icon" (click)="refresh.emit()" title="Refresh Graph">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="1 4 1 10 7 10"/>
          <polyline points="23 20 23 14 17 14"/>
          <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10M23 14l-4.64 4.36A9 9 0 0 1 3.51 15"/>
        </svg>
      </button>
    </div>
  `,
  styles: [`
    .graph-controls {
      display: flex;
      flex-direction: column;
      gap: var(--spacing-xs);
      padding: var(--spacing-sm);

      .btn--icon {
        width: 32px;
        height: 32px;

        svg {
          width: 18px;
          height: 18px;
        }
      }

      .divider {
        height: 1px;
        background: var(--glass-border);
        margin: var(--spacing-xs) 0;
      }
    }
  `],
})
export class GraphControlsComponent {
  @Output() zoomIn = new EventEmitter<void>();
  @Output() zoomOut = new EventEmitter<void>();
  @Output() resetView = new EventEmitter<void>();
  @Output() refresh = new EventEmitter<void>();
}
