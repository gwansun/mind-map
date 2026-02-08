import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
  selector: 'app-empty-state',
  standalone: true,
  template: `
    <div class="empty-state">
      <div class="empty-state__icon">
        @if (icon === 'graph') {
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="5" r="3"/>
            <circle cx="5" cy="19" r="3"/>
            <circle cx="19" cy="19" r="3"/>
            <path d="M12 8v3M9.5 14.5L7 16.5M14.5 14.5L17 16.5"/>
          </svg>
        } @else if (icon === 'chat') {
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
          </svg>
        } @else {
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
            <polyline points="13 2 13 9 20 9"/>
          </svg>
        }
      </div>
      <h3 class="empty-state__title">{{ title }}</h3>
      @if (description) {
        <p class="empty-state__description">{{ description }}</p>
      }
      @if (actionLabel) {
        <button class="btn btn--primary" (click)="action.emit()">{{ actionLabel }}</button>
      }
    </div>
  `,
  styles: [`
    .empty-state {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: var(--spacing-2xl);
      gap: var(--spacing-md);
    }

    .empty-state__icon {
      width: 64px;
      height: 64px;
      color: var(--color-text-muted);
      opacity: 0.5;

      svg {
        width: 100%;
        height: 100%;
      }
    }

    .empty-state__title {
      font-size: 18px;
      font-weight: 600;
      color: var(--color-text-primary);
    }

    .empty-state__description {
      font-size: 14px;
      color: var(--color-text-secondary);
      max-width: 300px;
    }
  `],
})
export class EmptyStateComponent {
  @Input() icon: 'graph' | 'chat' | 'document' = 'document';
  @Input() title = 'Nothing here yet';
  @Input() description = '';
  @Input() actionLabel = '';
  @Output() action = new EventEmitter<void>();
}
