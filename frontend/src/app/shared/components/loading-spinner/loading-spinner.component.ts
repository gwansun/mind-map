import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-loading-spinner',
  standalone: true,
  template: `
    <div class="spinner-wrapper" [class.spinner-wrapper--overlay]="overlay">
      <div class="spinner" [style.width.px]="size" [style.height.px]="size"></div>
      @if (message) {
        <span class="spinner-message">{{ message }}</span>
      }
    </div>
  `,
  styles: [`
    .spinner-wrapper {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: var(--spacing-md);
      padding: var(--spacing-lg);

      &--overlay {
        position: absolute;
        inset: 0;
        background: rgba(10, 10, 15, 0.8);
        backdrop-filter: blur(4px);
        z-index: 100;
      }
    }

    .spinner {
      border: 3px solid var(--color-bg-tertiary);
      border-top-color: var(--color-accent-primary);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    .spinner-message {
      color: var(--color-text-secondary);
      font-size: 14px;
    }
  `],
})
export class LoadingSpinnerComponent {
  @Input() size = 32;
  @Input() message = '';
  @Input() overlay = false;
}
