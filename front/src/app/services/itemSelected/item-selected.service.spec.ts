import { TestBed } from '@angular/core/testing';

import { ItemSelectedService } from './item-selected.service';

describe('ItemSelectedService', () => {
  let service: ItemSelectedService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ItemSelectedService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
