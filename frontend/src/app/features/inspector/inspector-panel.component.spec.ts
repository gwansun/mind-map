import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';

import { InspectorPanelComponent } from './inspector-panel.component';
import { ApiService } from '../../core';
import { D3Node, Edge, NodeType } from '../../models';

// --- Test helpers ---

function makeD3Node(overrides: Partial<D3Node> = {}): D3Node {
  return {
    id: 'node_test',
    document: 'Test concept node',
    metadata: {
      type: NodeType.CONCEPT,
      created_at: 1700000000,
      last_interaction: 1700000000,
      connection_count: 0,
      importance_score: 0.5,
    },
    ...overrides,
  };
}

function makeNodeDetailResponse(node: D3Node, edges: Edge[] = []) {
  return {
    node: node,
    edges,
    importance_score: node.metadata.importance_score,
  };
}

// --- Mock ApiService ---

class MockApiService {
  getNode = jasmine.createSpy('getNode').and.returnValue({
    toPromise: jasmine.createSpy('toPromise').and.resolveTo(null),
  });
  deleteNode = jasmine.createSpy('deleteNode').and.returnValue({
    toPromise: jasmine.createSpy('toPromise').and.resolveTo({
      node_id: 'node_test',
      deleted_tag_ids: [],
      deleted_edges_count: 0,
    }),
  });
}

describe('InspectorPanelComponent', () => {
  let component: InspectorPanelComponent;
  let fixture: ComponentFixture<InspectorPanelComponent>;
  let apiService: MockApiService;

  async function setNodeAndWait(node: D3Node, edges: Edge[] = []): Promise<void> {
    const response = makeNodeDetailResponse(node, edges);
    apiService.getNode.and.returnValue({
      toPromise: jasmine.createSpy('toPromise').and.resolveTo(response),
    });

    component.node = node;
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
  }

  beforeEach(async () => {
    apiService = new MockApiService();

    await TestBed.configureTestingModule({
      imports: [InspectorPanelComponent],
      providers: [{ provide: ApiService, useValue: apiService }],
    }).compileComponents();

    fixture = TestBed.createComponent(InspectorPanelComponent);
    component = fixture.componentInstance;
  });

  describe('initial state', () => {
    it('should show empty state when no node is selected', () => {
      component.node = null;
      fixture.detectChanges();

      const emptyText = fixture.debugElement.query(By.css('.inspector-empty'));
      expect(emptyText).toBeTruthy();
      expect(emptyText.nativeElement.textContent).toContain('Select a node to view details');
    });

    it('should not show delete button when no node is selected', () => {
      component.node = null;
      fixture.detectChanges();
      const deleteBtn = fixture.debugElement.query(By.css('.btn--danger'));
      expect(deleteBtn).toBeNull();
    });
  });

  describe('node loaded', () => {
    beforeEach(async () => {
      const node = makeD3Node();
      await setNodeAndWait(node, []);
    });

    it('should show delete button once node is loaded', () => {
      const deleteBtn = fixture.debugElement.query(By.css('button[title="Delete node"]'));
      expect(deleteBtn).toBeTruthy();
    });

    it('should display node type badge', () => {
      const badge = fixture.debugElement.query(By.css('.node-type-badge'));
      expect(badge.nativeElement.textContent.trim()).toBe('Concept');
    });

    it('should display node document', () => {
      const docEl = fixture.debugElement.query(By.css('.node-document'));
      expect(docEl.nativeElement.textContent).toContain('Test concept node');
    });
  });

  describe('delete flow', () => {
    let node: D3Node;

    beforeEach(async () => {
      node = makeD3Node({ id: 'node_delete_test', document: 'Node to delete' });
      await setNodeAndWait(node, [
        { source: 'node_delete_test', target: 'neighbor_1', weight: 1.0, relation_type: 'related_to' },
        { source: 'neighbor_2', target: 'node_delete_test', weight: 0.8, relation_type: 'mentions' },
      ]);
    });

    function openDeleteConfirm(): void {
      const deleteBtn = fixture.debugElement.query(By.css('button[title="Delete node"]'));
      deleteBtn.nativeElement.click();
      fixture.detectChanges();
    }

    it('should show confirmation dialog when delete button is clicked', () => {
      openDeleteConfirm();

      const confirmArea = fixture.debugElement.query(By.css('.delete-confirm'));
      expect(confirmArea).toBeTruthy();
      expect(confirmArea.nativeElement.textContent).toContain('Node to delete');
      expect(confirmArea.nativeElement.textContent).toContain('2 connections');
    });

    it('should hide confirmation when cancel is clicked', () => {
      openDeleteConfirm();

      const cancelBtn = fixture.debugElement.query(By.css('.delete-confirm .btn--ghost'));
      cancelBtn.nativeElement.click();
      fixture.detectChanges();

      const confirmArea = fixture.debugElement.query(By.css('.delete-confirm'));
      expect(confirmArea).toBeNull();
    });

    it('should call API and emit deleted on successful delete', async () => {
      openDeleteConfirm();

      const confirmDeleteBtn = fixture.debugElement.query(By.css('.delete-confirm .btn--danger'));
      let deletedEmitted = false;
      component.deleted.subscribe((id) => {
        expect(id).toBe('node_delete_test');
        deletedEmitted = true;
      });

      confirmDeleteBtn.nativeElement.click();
      fixture.detectChanges();

      expect(apiService.deleteNode).toHaveBeenCalledWith('node_delete_test');
      await fixture.whenStable();
      expect(deletedEmitted).toBeTrue();
    });

    it('should show Deleting... text while deleting', () => {
      apiService.deleteNode.and.returnValue({
        toPromise: jasmine.createSpy('toPromise').and.returnValue(new Promise(() => {})),
      });

      openDeleteConfirm();

      const confirmDeleteBtn = fixture.debugElement.query(By.css('.delete-confirm .btn--danger'));
      confirmDeleteBtn.nativeElement.click();
      fixture.detectChanges();

      expect(component.isDeleting()).toBeTrue();
      const deleteText = fixture.debugElement.query(By.css('.delete-confirm .btn--danger'));
      expect(deleteText.nativeElement.textContent).toContain('Deleting...');
    });

    it('should truncate long documents in confirmation dialog', async () => {
      const longNode = makeD3Node({
        id: 'node_long',
        document: 'This is a very long document that exceeds fifty characters and should be truncated',
      });
      await setNodeAndWait(longNode, []);

      openDeleteConfirm();

      const confirmArea = fixture.debugElement.query(By.css('.delete-confirm'));
      expect(confirmArea.nativeElement.textContent).toContain('…');
      expect(confirmArea.nativeElement.textContent).not.toContain(
        'This is a very long document that exceeds fifty characters and should be truncated'
      );
    });
  });
});
