import { Component, OnInit, ViewChild, inject, ChangeDetectionStrategy } from '@angular/core';
import { GraphService } from './graph.service';
import { GraphCanvasComponent } from './components/graph-canvas/graph-canvas.component';
import { GraphControlsComponent } from './components/graph-controls/graph-controls.component';
import { NodeSearchComponent } from './components/node-search/node-search.component';
import { LoadingSpinnerComponent, EmptyStateComponent } from '../../shared';
import { ApiService } from '../../core';
import { D3Node } from '../../models';

@Component({
  selector: 'app-graph-container',
  standalone: true,
  imports: [
    GraphCanvasComponent,
    GraphControlsComponent,
    NodeSearchComponent,
    LoadingSpinnerComponent,
    EmptyStateComponent,
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="graph-container">
      @if (apiService.isLoadingGraph()) {
        <app-loading-spinner [overlay]="true" message="Loading knowledge graph..."></app-loading-spinner>
      }

      @if (graphService.nodes().length === 0 && !apiService.isLoadingGraph()) {
        <app-empty-state
          icon="graph"
          title="No knowledge yet"
          description="Add your first memo to start building your knowledge graph."
        ></app-empty-state>
      } @else {
        <app-graph-canvas
          #canvas
          [nodes]="graphService.nodes()"
          [links]="graphService.links()"
          [selectedNodeId]="graphService.selectedNodeId()"
          [hoveredNodeId]="graphService.hoveredNodeId()"
          (nodeClick)="onNodeClick($event)"
          (nodeHover)="onNodeHover($event)"
          (canvasClick)="onCanvasClick()"
        ></app-graph-canvas>
      }

      <div class="graph-overlay-top">
        <app-node-search
          [nodes]="graphService.nodes()"
          (nodeSelect)="onNodeSelect($event)"
          (queryChange)="graphService.setSearchQuery($event)"
        ></app-node-search>

        <div class="graph-stats glass-panel">
          <span class="stat">
            <strong>{{ graphService.nodeCount() }}</strong> nodes
          </span>
          <span class="divider">|</span>
          <span class="stat">
            <strong>{{ graphService.edgeCount() }}</strong> edges
          </span>
        </div>
      </div>

      <div class="graph-overlay-bottom-left">
        <app-graph-controls
          (zoomIn)="canvas?.zoomIn()"
          (zoomOut)="canvas?.zoomOut()"
          (resetView)="canvas?.resetZoom()"
          (refresh)="refreshGraph()"
        ></app-graph-controls>
      </div>
    </div>
  `,
  styles: [`
    .graph-container {
      position: relative;
      width: 100%;
      height: 100%;
      overflow: hidden;
    }

    .graph-overlay-top {
      position: absolute;
      top: var(--spacing-md);
      left: var(--spacing-md);
      right: var(--spacing-md);
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      pointer-events: none;

      > * {
        pointer-events: auto;
      }
    }

    .graph-overlay-bottom-left {
      position: absolute;
      bottom: var(--spacing-md);
      left: var(--spacing-md);
    }

    .graph-stats {
      display: flex;
      align-items: center;
      gap: var(--spacing-sm);
      padding: var(--spacing-sm) var(--spacing-md);
      font-size: 12px;
      color: var(--color-text-secondary);

      strong {
        color: var(--color-text-primary);
      }

      .divider {
        color: var(--color-text-muted);
      }
    }
  `],
})
export class GraphContainerComponent implements OnInit {
  @ViewChild('canvas') canvas?: GraphCanvasComponent;

  protected graphService = inject(GraphService);
  protected apiService = inject(ApiService);

  ngOnInit(): void {
    this.graphService.loadGraph();
  }

  onNodeClick(node: D3Node): void {
    this.graphService.selectNode(node.id);
  }

  onNodeHover(node: D3Node | null): void {
    this.graphService.hoverNode(node?.id ?? null);
  }

  onCanvasClick(): void {
    this.graphService.clearSelection();
  }

  onNodeSelect(node: D3Node): void {
    this.graphService.selectNode(node.id);
    this.canvas?.centerOnNode(node.id);
  }

  refreshGraph(): void {
    this.graphService.loadGraph(true);
  }
}
