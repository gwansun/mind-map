import { Injectable, signal, inject } from '@angular/core';
import { ApiService } from '../../core';

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: number;
  contextNodes?: string[];
}

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  private apiService = inject(ApiService);

  readonly messages = signal<ChatMessage[]>([]);
  readonly isLoading = signal(false);

  /**
   * Send a question to the knowledge graph
   */
  async ask(query: string): Promise<void> {
    if (!query.trim()) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      type: 'user',
      content: query,
      timestamp: Date.now(),
    };
    this.messages.update((msgs) => [...msgs, userMessage]);

    this.isLoading.set(true);

    try {
      const response = await this.apiService.ask({ query }).toPromise();

      if (response) {
        const assistantMessage: ChatMessage = {
          id: crypto.randomUUID(),
          type: 'assistant',
          content: response.response,
          timestamp: Date.now(),
          contextNodes: response.context_nodes,
        };
        this.messages.update((msgs) => [...msgs, assistantMessage]);
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
        type: 'assistant',
        content: 'Sorry, I encountered an error while processing your question. Please try again.',
        timestamp: Date.now(),
      };
      this.messages.update((msgs) => [...msgs, errorMessage]);
    } finally {
      this.isLoading.set(false);
    }
  }

  /**
   * Add a memo to the knowledge graph
   */
  async addMemo(text: string, source?: string): Promise<boolean> {
    if (!text.trim()) return false;

    this.isLoading.set(true);

    try {
      const response = await this.apiService.addMemo({ text, source }).toPromise();
      return response?.status === 'success';
    } catch (error) {
      return false;
    } finally {
      this.isLoading.set(false);
    }
  }

  /**
   * Clear chat history
   */
  clearMessages(): void {
    this.messages.set([]);
  }
}
