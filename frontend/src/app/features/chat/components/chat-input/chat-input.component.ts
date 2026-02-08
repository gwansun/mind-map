import {
  Component,
  Input,
  Output,
  EventEmitter,
  signal,
  ChangeDetectionStrategy,
  ElementRef,
  ViewChild,
} from '@angular/core';
import { FormsModule } from '@angular/forms';

export type InputMode = 'ask' | 'memo';

@Component({
  selector: 'app-chat-input',
  standalone: true,
  imports: [FormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="chat-input">
      <div class="mode-toggle">
        <button
          class="mode-btn"
          [class.active]="mode() === 'ask'"
          (click)="mode.set('ask')"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          Ask
        </button>
        <button
          class="mode-btn"
          [class.active]="mode() === 'memo'"
          (click)="mode.set('memo')"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="12" y1="18" x2="12" y2="12"/>
            <line x1="9" y1="15" x2="15" y2="15"/>
          </svg>
          Add Memo
        </button>
      </div>

      <div class="input-wrapper">
        <textarea
          #textarea
          class="input-textarea"
          [placeholder]="placeholder()"
          [(ngModel)]="inputValue"
          (keydown.enter)="onKeyDown($event)"
          [disabled]="disabled"
          rows="1"
        ></textarea>
        <button
          class="send-btn"
          [disabled]="disabled || !inputValue.trim()"
          (click)="onSubmit()"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>

      <div class="input-hint">
        Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line
      </div>
    </div>
  `,
  styles: [`
    .chat-input {
      padding: var(--spacing-md);
      border-top: 1px solid var(--glass-border);
      background: var(--color-bg-secondary);
    }

    .mode-toggle {
      display: flex;
      gap: var(--spacing-sm);
      margin-bottom: var(--spacing-sm);
    }

    .mode-btn {
      display: flex;
      align-items: center;
      gap: var(--spacing-xs);
      padding: var(--spacing-xs) var(--spacing-sm);
      background: none;
      border: 1px solid var(--glass-border);
      border-radius: var(--radius-md);
      color: var(--color-text-secondary);
      font-size: 12px;
      cursor: pointer;
      transition: all var(--transition-fast);

      svg {
        width: 14px;
        height: 14px;
      }

      &:hover {
        background: var(--color-bg-tertiary);
        color: var(--color-text-primary);
      }

      &.active {
        background: var(--color-accent-primary);
        border-color: var(--color-accent-primary);
        color: white;
      }
    }

    .input-wrapper {
      display: flex;
      gap: var(--spacing-sm);
      align-items: flex-end;
    }

    .input-textarea {
      flex: 1;
      min-height: 40px;
      max-height: 120px;
      padding: var(--spacing-sm) var(--spacing-md);
      background: var(--color-bg-tertiary);
      border: 1px solid var(--glass-border);
      border-radius: var(--radius-md);
      color: var(--color-text-primary);
      font-family: inherit;
      font-size: 14px;
      line-height: 1.5;
      resize: none;
      outline: none;
      transition: border-color var(--transition-fast);

      &:focus {
        border-color: var(--color-accent-primary);
      }

      &::placeholder {
        color: var(--color-text-muted);
      }

      &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
    }

    .send-btn {
      width: 40px;
      height: 40px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: var(--color-accent-primary);
      border: none;
      border-radius: var(--radius-md);
      color: white;
      cursor: pointer;
      transition: all var(--transition-fast);

      svg {
        width: 18px;
        height: 18px;
      }

      &:hover:not(:disabled) {
        background: var(--color-accent-secondary);
      }

      &:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
    }

    .input-hint {
      margin-top: var(--spacing-xs);
      font-size: 11px;
      color: var(--color-text-muted);
      text-align: center;

      kbd {
        padding: 2px 4px;
        background: var(--color-bg-tertiary);
        border-radius: var(--radius-sm);
        font-family: var(--font-mono);
        font-size: 10px;
      }
    }
  `],
})
export class ChatInputComponent {
  @ViewChild('textarea') textarea!: ElementRef<HTMLTextAreaElement>;

  @Input() disabled = false;

  @Output() ask = new EventEmitter<string>();
  @Output() memo = new EventEmitter<string>();

  readonly mode = signal<InputMode>('ask');
  inputValue = '';

  readonly placeholder = () => {
    return this.mode() === 'ask'
      ? 'Ask a question about your knowledge...'
      : 'Add a memo to your knowledge graph...';
  };

  onKeyDown(event: Event): void {
    const keyEvent = event as KeyboardEvent;
    if (keyEvent.key === 'Enter' && !keyEvent.shiftKey) {
      event.preventDefault();
      this.onSubmit();
    }
  }

  onSubmit(): void {
    const value = this.inputValue.trim();
    if (!value) return;

    if (this.mode() === 'ask') {
      this.ask.emit(value);
    } else {
      this.memo.emit(value);
    }

    this.inputValue = '';
    this.autoResize();
  }

  private autoResize(): void {
    const textarea = this.textarea?.nativeElement;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
  }
}
