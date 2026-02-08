import {
  Component,
  Input,
  Output,
  EventEmitter,
  ElementRef,
  ViewChild,
  AfterViewChecked,
  ChangeDetectionStrategy,
} from '@angular/core';
import { DatePipe } from '@angular/common';
import { ChatMessage } from '../../chat.service';
import { MarkdownPipe } from './markdown.pipe';

@Component({
  selector: 'app-message-list',
  standalone: true,
  imports: [DatePipe, MarkdownPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="message-list" #scrollContainer>
      @for (message of messages; track message.id) {
        <div class="message" [class]="'message--' + message.type">
          <div class="message__header">
            <span class="message__author">
              {{ message.type === 'user' ? 'You' : 'Mind Map' }}
            </span>
            <span class="message__time">
              {{ message.timestamp | date:'shortTime' }}
            </span>
          </div>
          <div class="message__content" [innerHTML]="message.content | markdown"></div>
          @if (message.contextNodes?.length) {
            <div class="message__context">
              <span class="context-label">Referenced nodes:</span>
              @for (nodeId of message.contextNodes!.slice(0, 3); track nodeId) {
                <button class="context-node" (click)="nodeClick.emit(nodeId)">
                  {{ nodeId.substring(0, 8) }}...
                </button>
              }
              @if (message.contextNodes!.length > 3) {
                <span class="context-more">+{{ message.contextNodes!.length - 3 }} more</span>
              }
            </div>
          }
        </div>
      }
      @if (isLoading) {
        <div class="message message--assistant message--loading">
          <div class="message__header">
            <span class="message__author">Mind Map</span>
          </div>
          <div class="message__content">
            <span class="typing-indicator">
              <span></span><span></span><span></span>
            </span>
          </div>
        </div>
      }
    </div>
  `,
  styles: [`
    .message-list {
      flex: 1;
      overflow-y: auto;
      padding: var(--spacing-md);
      display: flex;
      flex-direction: column;
      gap: var(--spacing-md);
    }

    .message {
      max-width: 85%;
      animation: slideUp var(--transition-normal);

      &--user {
        align-self: flex-end;

        .message__content {
          background: var(--color-accent-primary);
          color: white;
          border-radius: var(--radius-lg) var(--radius-lg) var(--radius-sm) var(--radius-lg);
        }
      }

      &--assistant {
        align-self: flex-start;

        .message__content {
          background: var(--color-bg-tertiary);
          border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) var(--radius-sm);
        }
      }
    }

    .message__header {
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
      margin-bottom: var(--spacing-xs);
      padding: 0 var(--spacing-sm);
    }

    .message__author {
      font-size: 12px;
      font-weight: 600;
      color: var(--color-text-secondary);
    }

    .message__time {
      font-size: 11px;
      color: var(--color-text-muted);
    }

    .message__content {
      padding: var(--spacing-sm) var(--spacing-md);
      font-size: 14px;
      line-height: 1.6;

      :host ::ng-deep {
        p {
          margin: 0 0 var(--spacing-sm) 0;

          &:last-child {
            margin-bottom: 0;
          }
        }

        code {
          font-family: var(--font-mono);
          font-size: 0.9em;
          background: rgba(0, 0, 0, 0.2);
          padding: 2px 6px;
          border-radius: var(--radius-sm);
        }

        pre {
          margin: var(--spacing-sm) 0;
          padding: var(--spacing-sm);
          background: rgba(0, 0, 0, 0.2);
          border-radius: var(--radius-md);
          overflow-x: auto;

          code {
            background: none;
            padding: 0;
          }
        }

        ul, ol {
          margin: var(--spacing-sm) 0;
          padding-left: var(--spacing-lg);
        }

        li {
          margin-bottom: var(--spacing-xs);
        }
      }
    }

    .message__context {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: var(--spacing-xs);
      margin-top: var(--spacing-sm);
      padding: 0 var(--spacing-sm);
    }

    .context-label {
      font-size: 11px;
      color: var(--color-text-muted);
    }

    .context-node {
      font-size: 10px;
      font-family: var(--font-mono);
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

    .context-more {
      font-size: 11px;
      color: var(--color-text-muted);
    }

    .typing-indicator {
      display: flex;
      gap: 4px;
      padding: var(--spacing-xs);

      span {
        width: 8px;
        height: 8px;
        background: var(--color-text-muted);
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;

        &:nth-child(1) { animation-delay: -0.32s; }
        &:nth-child(2) { animation-delay: -0.16s; }
      }
    }

    @keyframes bounce {
      0%, 80%, 100% { transform: scale(0); }
      40% { transform: scale(1); }
    }
  `],
})
export class MessageListComponent implements AfterViewChecked {
  @ViewChild('scrollContainer') scrollContainer!: ElementRef<HTMLDivElement>;

  @Input() messages: ChatMessage[] = [];
  @Input() isLoading = false;

  @Output() nodeClick = new EventEmitter<string>();

  private shouldScrollToBottom = true;

  ngAfterViewChecked(): void {
    if (this.shouldScrollToBottom) {
      this.scrollToBottom();
    }
  }

  private scrollToBottom(): void {
    const container = this.scrollContainer?.nativeElement;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }
}
