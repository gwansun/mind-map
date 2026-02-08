import { Component, ViewChild, inject, ChangeDetectionStrategy } from '@angular/core';
import { GraphContainerComponent, GraphService, GraphCanvasComponent } from './features/graph';
import { ChatContainerComponent } from './features/chat';
import { InspectorPanelComponent } from './features/inspector';
import { ToastComponent } from './shared';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    GraphContainerComponent,
    ChatContainerComponent,
    InspectorPanelComponent,
    ToastComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="app-layout" [class.inspector-open]="graphService.selectedNode()">
      <!-- Main Graph Area -->
      <main class="main-area">
        <app-graph-container></app-graph-container>
      </main>

      <!-- Chat Overlay -->
      <aside class="chat-sidebar">
        <app-chat-container
          (selectNode)="onSelectNode($event)"
        ></app-chat-container>
      </aside>

      <!-- Inspector Panel (conditional) -->
      @if (graphService.selectedNode(); as selectedNode) {
        <aside class="inspector-sidebar">
          <app-inspector-panel
            [node]="selectedNode"
            (close)="graphService.clearSelection()"
            (selectNode)="onSelectNode($event)"
          ></app-inspector-panel>
        </aside>
      }
    </div>

    <!-- Toast notifications -->
    <app-toast></app-toast>
  `,
  styles: [`
    .app-layout {
      display: grid;
      grid-template-columns: 1fr 380px;
      grid-template-rows: 1fr;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
      gap: var(--spacing-md);
      padding: var(--spacing-md);
      transition: grid-template-columns var(--transition-normal);

      &.inspector-open {
        grid-template-columns: 1fr 380px 320px;
      }
    }

    .main-area {
      min-width: 0;
      border-radius: var(--radius-lg);
      overflow: hidden;
      background: var(--color-bg-secondary);
      border: 1px solid var(--glass-border);
    }

    .chat-sidebar {
      min-width: 0;
      height: 100%;
    }

    .inspector-sidebar {
      min-width: 0;
      height: 100%;
      animation: slideIn var(--transition-normal);
    }

    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateX(20px);
      }
      to {
        opacity: 1;
        transform: translateX(0);
      }
    }

    /* Responsive: Tablet */
    @media (max-width: 1024px) {
      .app-layout {
        grid-template-columns: 1fr 320px;

        &.inspector-open {
          grid-template-columns: 1fr 320px;

          .inspector-sidebar {
            position: fixed;
            right: var(--spacing-md);
            top: var(--spacing-md);
            bottom: var(--spacing-md);
            width: 300px;
            z-index: 100;
          }
        }
      }
    }

    /* Responsive: Mobile */
    @media (max-width: 768px) {
      .app-layout {
        grid-template-columns: 1fr;
        grid-template-rows: 1fr auto;

        &.inspector-open {
          grid-template-columns: 1fr;
        }
      }

      .main-area {
        grid-row: 1;
      }

      .chat-sidebar {
        grid-row: 2;
        height: 300px;
        max-height: 40vh;
      }

      .inspector-sidebar {
        position: fixed;
        inset: var(--spacing-md);
        z-index: 100;
      }
    }
  `],
})
export class AppComponent {
  @ViewChild(GraphContainerComponent) graphContainer?: GraphContainerComponent;

  protected graphService = inject(GraphService);

  onSelectNode(nodeId: string): void {
    this.graphService.selectNode(nodeId);
    // Center on the node in the graph
    this.graphContainer?.canvas?.centerOnNode(nodeId);
  }
}
