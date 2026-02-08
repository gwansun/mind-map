import {
  Component,
  ElementRef,
  Input,
  Output,
  EventEmitter,
  ViewChild,
  AfterViewInit,
  OnDestroy,
  OnChanges,
  SimpleChanges,
  ChangeDetectionStrategy,
} from '@angular/core';
import * as d3 from 'd3';
import { D3Node, D3Link, NodeType } from '../../../../models';

@Component({
  selector: 'app-graph-canvas',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="graph-canvas" #container>
      <svg #svg></svg>
    </div>
  `,
  styles: [`
    .graph-canvas {
      width: 100%;
      height: 100%;
      position: relative;
      overflow: hidden;
      background: var(--color-bg-primary);
    }

    svg {
      width: 100%;
      height: 100%;
    }

    :host ::ng-deep {
      .node {
        cursor: pointer;
        transition: opacity 0.2s;

        &:hover {
          opacity: 0.8;
        }

        &.selected {
          stroke: var(--color-accent-tertiary);
          stroke-width: 3px;
        }

        &.connected {
          opacity: 1 !important;
        }

        &.dimmed {
          opacity: 0.2;
        }
      }

      .node-label {
        font-size: 10px;
        fill: var(--color-text-primary);
        pointer-events: none;
        text-anchor: middle;
        dominant-baseline: middle;
      }

      .link {
        stroke: var(--color-edge-default);
        stroke-opacity: 0.6;
        transition: stroke 0.2s, stroke-opacity 0.2s;

        &.highlighted {
          stroke: var(--color-edge-hover);
          stroke-opacity: 1;
        }

        &.dimmed {
          stroke-opacity: 0.1;
        }
      }
    }
  `],
})
export class GraphCanvasComponent implements AfterViewInit, OnDestroy, OnChanges {
  @ViewChild('container') containerRef!: ElementRef<HTMLDivElement>;
  @ViewChild('svg') svgRef!: ElementRef<SVGSVGElement>;

  @Input() nodes: D3Node[] = [];
  @Input() links: D3Link[] = [];
  @Input() selectedNodeId: string | null = null;
  @Input() hoveredNodeId: string | null = null;

  @Output() nodeClick = new EventEmitter<D3Node>();
  @Output() nodeHover = new EventEmitter<D3Node | null>();
  @Output() canvasClick = new EventEmitter<void>();

  private svg!: d3.Selection<SVGSVGElement, unknown, null, undefined>;
  private g!: d3.Selection<SVGGElement, unknown, null, undefined>;
  private simulation!: d3.Simulation<D3Node, D3Link>;
  private zoom!: d3.ZoomBehavior<SVGSVGElement, unknown>;
  private resizeObserver!: ResizeObserver;

  private linkSelection!: d3.Selection<SVGLineElement, D3Link, SVGGElement, unknown>;
  private nodeSelection!: d3.Selection<SVGCircleElement, D3Node, SVGGElement, unknown>;
  private labelSelection!: d3.Selection<SVGTextElement, D3Node, SVGGElement, unknown>;

  ngAfterViewInit(): void {
    this.initializeSvg();
    this.initializeZoom();
    this.initializeSimulation();
    this.setupResizeObserver();
    this.render();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (this.svg && (changes['nodes'] || changes['links'])) {
      this.render();
    }
    if (this.svg && (changes['selectedNodeId'] || changes['hoveredNodeId'])) {
      this.updateHighlighting();
    }
  }

  ngOnDestroy(): void {
    this.simulation?.stop();
    this.resizeObserver?.disconnect();
  }

  private initializeSvg(): void {
    this.svg = d3.select(this.svgRef.nativeElement);
    this.g = this.svg.append('g');

    // Click on canvas to deselect
    this.svg.on('click', (event) => {
      if (event.target === this.svgRef.nativeElement) {
        this.canvasClick.emit();
      }
    });
  }

  private initializeZoom(): void {
    this.zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        this.g.attr('transform', event.transform);
      });

    this.svg.call(this.zoom);
  }

  private initializeSimulation(): void {
    const { width, height } = this.getContainerSize();

    this.simulation = d3
      .forceSimulation<D3Node, D3Link>()
      .force('link', d3.forceLink<D3Node, D3Link>().id((d) => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d) => this.getNodeSize(d as D3Node) + 10))
      .on('tick', () => this.tick());
  }

  private setupResizeObserver(): void {
    this.resizeObserver = new ResizeObserver(() => {
      const { width, height } = this.getContainerSize();
      this.simulation.force('center', d3.forceCenter(width / 2, height / 2));
      this.simulation.alpha(0.3).restart();
    });
    this.resizeObserver.observe(this.containerRef.nativeElement);
  }

  private render(): void {
    // Links
    this.linkSelection = this.g
      .selectAll<SVGLineElement, D3Link>('.link')
      .data(this.links, (d) => `${(d.source as D3Node).id || d.source}-${(d.target as D3Node).id || d.target}`)
      .join('line')
      .attr('class', 'link')
      .attr('stroke-width', (d) => Math.sqrt(d.weight) * 2);

    // Nodes
    this.nodeSelection = this.g
      .selectAll<SVGCircleElement, D3Node>('.node')
      .data(this.nodes, (d) => d.id)
      .join('circle')
      .attr('class', 'node')
      .attr('r', (d) => this.getNodeSize(d))
      .attr('fill', (d) => this.getNodeColor(d))
      .call(this.drag())
      .on('click', (event, d) => {
        event.stopPropagation();
        this.nodeClick.emit(d);
      })
      .on('mouseenter', (_, d) => this.nodeHover.emit(d))
      .on('mouseleave', () => this.nodeHover.emit(null));

    // Labels
    this.labelSelection = this.g
      .selectAll<SVGTextElement, D3Node>('.node-label')
      .data(this.nodes.filter((n) => n.metadata.type !== NodeType.TAG), (d) => d.id)
      .join('text')
      .attr('class', 'node-label')
      .attr('dy', (d) => this.getNodeSize(d) + 14)
      .text((d) => this.truncateLabel(d.document, 20));

    // Update simulation
    this.simulation.nodes(this.nodes);
    (this.simulation.force('link') as d3.ForceLink<D3Node, D3Link>).links(this.links);
    this.simulation.alpha(1).restart();

    this.updateHighlighting();
  }

  private tick(): void {
    this.linkSelection
      ?.attr('x1', (d) => (d.source as D3Node).x!)
      .attr('y1', (d) => (d.source as D3Node).y!)
      .attr('x2', (d) => (d.target as D3Node).x!)
      .attr('y2', (d) => (d.target as D3Node).y!);

    this.nodeSelection?.attr('cx', (d) => d.x!).attr('cy', (d) => d.y!);
    this.labelSelection?.attr('x', (d) => d.x!).attr('y', (d) => d.y!);
  }

  private drag(): d3.DragBehavior<SVGCircleElement, D3Node, D3Node | d3.SubjectPosition> {
    return d3
      .drag<SVGCircleElement, D3Node>()
      .on('start', (event, d) => {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
  }

  private updateHighlighting(): void {
    const selectedId = this.selectedNodeId;
    const hoveredId = this.hoveredNodeId;
    const highlightId = hoveredId || selectedId;

    if (!highlightId) {
      this.nodeSelection?.classed('selected', false).classed('connected', false).classed('dimmed', false);
      this.linkSelection?.classed('highlighted', false).classed('dimmed', false);
      return;
    }

    // Find connected nodes
    const connectedIds = new Set<string>([highlightId]);
    this.links.forEach((link) => {
      const sourceId = typeof link.source === 'string' ? link.source : (link.source as D3Node).id;
      const targetId = typeof link.target === 'string' ? link.target : (link.target as D3Node).id;
      if (sourceId === highlightId) connectedIds.add(targetId);
      if (targetId === highlightId) connectedIds.add(sourceId);
    });

    // Update nodes
    this.nodeSelection
      ?.classed('selected', (d) => d.id === selectedId)
      .classed('connected', (d) => connectedIds.has(d.id))
      .classed('dimmed', (d) => !connectedIds.has(d.id));

    // Update links
    this.linkSelection
      ?.classed('highlighted', (d) => {
        const sourceId = typeof d.source === 'string' ? d.source : (d.source as D3Node).id;
        const targetId = typeof d.target === 'string' ? d.target : (d.target as D3Node).id;
        return sourceId === highlightId || targetId === highlightId;
      })
      .classed('dimmed', (d) => {
        const sourceId = typeof d.source === 'string' ? d.source : (d.source as D3Node).id;
        const targetId = typeof d.target === 'string' ? d.target : (d.target as D3Node).id;
        return sourceId !== highlightId && targetId !== highlightId;
      });
  }

  private getNodeSize(node: D3Node): number {
    const baseSize = {
      [NodeType.CONCEPT]: 24,
      [NodeType.ENTITY]: 18,
      [NodeType.TAG]: 12,
    }[node.metadata.type] || 16;
    return baseSize * (1 + node.metadata.importance_score * 0.5);
  }

  private getNodeColor(node: D3Node): string {
    return {
      [NodeType.CONCEPT]: '#6366f1',
      [NodeType.ENTITY]: '#22c55e',
      [NodeType.TAG]: '#f59e0b',
    }[node.metadata.type] || '#6366f1';
  }

  private getContainerSize(): { width: number; height: number } {
    const rect = this.containerRef?.nativeElement?.getBoundingClientRect();
    return {
      width: rect?.width || 800,
      height: rect?.height || 600,
    };
  }

  private truncateLabel(text: string, maxLength: number): string {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  }

  // Public methods for external control
  zoomIn(): void {
    this.svg.transition().call(this.zoom.scaleBy, 1.3);
  }

  zoomOut(): void {
    this.svg.transition().call(this.zoom.scaleBy, 0.7);
  }

  resetZoom(): void {
    const { width, height } = this.getContainerSize();
    this.svg.transition().call(
      this.zoom.transform,
      d3.zoomIdentity.translate(width / 2, height / 2).scale(1).translate(-width / 2, -height / 2)
    );
  }

  centerOnNode(nodeId: string): void {
    const node = this.nodes.find((n) => n.id === nodeId);
    if (node && node.x !== undefined && node.y !== undefined) {
      const { width, height } = this.getContainerSize();
      this.svg.transition().call(
        this.zoom.transform,
        d3.zoomIdentity.translate(width / 2 - node.x, height / 2 - node.y)
      );
    }
  }
}
