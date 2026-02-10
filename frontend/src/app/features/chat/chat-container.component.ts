import { Component, Output, EventEmitter, inject, ChangeDetectionStrategy } from '@angular/core';
import { ChatService } from './chat.service';
import { MessageListComponent } from './components/message-list/message-list.component';
import { ChatInputComponent } from './components/chat-input/chat-input.component';
import { EmptyStateComponent, StatusIndicatorComponent } from '../../shared';
import { GraphService } from '../graph';
import { ToastService } from '../../shared/components/toast/toast.service';

@Component({
  selector: 'app-chat-container',
  standalone: true,
  imports: [
    MessageListComponent,
    ChatInputComponent,
    EmptyStateComponent,
    StatusIndicatorComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="chat-container glass-panel">
      <div class="chat-header">
        <h3>Chat with Knowledge</h3>
        <div class="chat-actions">
          @if (chatService.isLoading()) {
            <app-status-indicator status="thinking"></app-status-indicator>
          }
          @if (chatService.messages().length > 0) {
            <button class="btn btn--ghost btn--icon" (click)="clearChat()" title="Clear chat">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"/>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
              </svg>
            </button>
          }
        </div>
      </div>

      @if (chatService.messages().length === 0) {
        <app-empty-state
          icon="chat"
          title="Start a conversation"
          description="Ask questions about your knowledge graph or add new memos."
        ></app-empty-state>
      } @else {
        <app-message-list
          [messages]="chatService.messages()"
          [isLoading]="chatService.isLoading()"
          (nodeClick)="onNodeClick($event)"
        ></app-message-list>
      }

      <app-chat-input
        [disabled]="chatService.isLoading()"
        (ask)="onAsk($event)"
        (memo)="onMemo($event)"
      ></app-chat-input>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100%;
    }

    .chat-container {
      display: flex;
      flex-direction: column;
      height: 100%;
      overflow: hidden;
    }

    .chat-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: var(--spacing-md);
      border-bottom: 1px solid var(--glass-border);

      h3 {
        font-size: 16px;
        font-weight: 600;
        color: var(--color-text-primary);
      }
    }

    .chat-actions {
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
    }

    app-empty-state {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    app-message-list {
      flex: 1;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
  `],
})
export class ChatContainerComponent {
  @Output() selectNode = new EventEmitter<string>();

  protected chatService = inject(ChatService);
  private graphService = inject(GraphService);
  private toastService = inject(ToastService);

  async onAsk(query: string): Promise<void> {
    await this.chatService.ask(query);
    // Refresh graph after ask (new nodes may have been created)
    this.graphService.loadGraph(true);
  }

  async onMemo(text: string): Promise<void> {
    const success = await this.chatService.addMemo(text);
    if (success) {
      this.toastService.success('Memo added to knowledge graph');
      // Refresh graph to show new nodes
      this.graphService.loadGraph(true);
    } else {
      this.toastService.error('Failed to add memo');
    }
  }

  onNodeClick(nodeId: string): void {
    this.selectNode.emit(nodeId);
    this.graphService.selectNode(nodeId);
  }

  clearChat(): void {
    this.chatService.clearMessages();
  }
}
