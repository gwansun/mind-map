import { Component, Input } from '@angular/core';

export type StatusType = 'idle' | 'loading' | 'thinking' | 'processing' | 'ingesting' | 'success' | 'error';

@Component({
  selector: 'app-status-indicator',
  standalone: true,
  template: `
    <div class="status" [class]="'status--' + status">
      @if (status === 'loading' || status === 'thinking' || status === 'processing' || status === 'ingesting') {
        <span class="status__spinner"></span>
      } @else if (status === 'success') {
        <svg class="status__icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
        </svg>
      } @else if (status === 'error') {
        <svg class="status__icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
      }
      <span class="status__label">{{ label }}</span>
    </div>
  `,
  styles: [`
    .status {
      display: inline-flex;
      align-items: center;
      gap: var(--spacing-sm);
      padding: var(--spacing-xs) var(--spacing-sm);
      border-radius: var(--radius-full);
      font-size: 12px;
      font-weight: 500;

      &--idle {
        color: var(--color-text-muted);
      }

      &--loading, &--thinking, &--processing, &--ingesting {
        color: var(--color-accent-secondary);
      }

      &--success {
        color: var(--color-success);
      }

      &--error {
        color: var(--color-error);
      }
    }

    .status__spinner {
      width: 12px;
      height: 12px;
      border: 2px solid currentColor;
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    .status__icon {
      width: 14px;
      height: 14px;
    }

    .status__label {
      white-space: nowrap;
    }
  `],
})
export class StatusIndicatorComponent {
  @Input() status: StatusType = 'idle';

  get label(): string {
    switch (this.status) {
      case 'idle':
        return 'Ready';
      case 'loading':
        return 'Loading...';
      case 'thinking':
        return 'Thinking...';
      case 'processing':
        return 'Processing...';
      case 'ingesting':
        return 'Ingesting...';
      case 'success':
        return 'Done';
      case 'error':
        return 'Error';
      default:
        return '';
    }
  }
}
