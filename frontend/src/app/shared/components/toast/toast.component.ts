import { Component, inject } from '@angular/core';
import { ToastService, Toast } from './toast.service';

@Component({
  selector: 'app-toast',
  standalone: true,
  template: `
    <div class="toast-container">
      @for (toast of toastService.toasts(); track toast.id) {
        <div class="toast toast--{{ toast.type }}" (click)="toastService.dismiss(toast.id)">
          <span class="toast__icon">
            @switch (toast.type) {
              @case ('success') { <svg viewBox="0 0 24 24" fill="currentColor"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg> }
              @case ('error') { <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg> }
              @case ('warning') { <svg viewBox="0 0 24 24" fill="currentColor"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg> }
              @case ('info') { <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg> }
            }
          </span>
          <span class="toast__message">{{ toast.message }}</span>
          <button class="toast__close" (click)="$event.stopPropagation(); toastService.dismiss(toast.id)">
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"/></svg>
          </button>
        </div>
      }
    </div>
  `,
  styles: [`
    .toast-container {
      position: fixed;
      bottom: var(--spacing-lg);
      right: var(--spacing-lg);
      z-index: 9999;
      display: flex;
      flex-direction: column-reverse;
      gap: var(--spacing-sm);
      max-width: 400px;
    }

    .toast {
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
      padding: var(--spacing-md);
      border-radius: var(--radius-md);
      background: var(--color-bg-elevated);
      border: 1px solid var(--glass-border);
      box-shadow: var(--shadow-lg);
      cursor: pointer;
      animation: slideUp var(--transition-normal);

      &--success {
        border-left: 3px solid var(--color-success);
        .toast__icon { color: var(--color-success); }
      }

      &--error {
        border-left: 3px solid var(--color-error);
        .toast__icon { color: var(--color-error); }
      }

      &--warning {
        border-left: 3px solid var(--color-warning);
        .toast__icon { color: var(--color-warning); }
      }

      &--info {
        border-left: 3px solid var(--color-info);
        .toast__icon { color: var(--color-info); }
      }
    }

    .toast__icon {
      flex-shrink: 0;
      width: 20px;
      height: 20px;

      svg {
        width: 100%;
        height: 100%;
      }
    }

    .toast__message {
      flex: 1;
      font-size: 14px;
      color: var(--color-text-primary);
    }

    .toast__close {
      flex-shrink: 0;
      width: 20px;
      height: 20px;
      padding: 0;
      background: none;
      border: none;
      color: var(--color-text-muted);
      cursor: pointer;
      transition: color var(--transition-fast);

      &:hover {
        color: var(--color-text-primary);
      }

      svg {
        width: 100%;
        height: 100%;
      }
    }
  `],
})
export class ToastComponent {
  protected toastService = inject(ToastService);
}
