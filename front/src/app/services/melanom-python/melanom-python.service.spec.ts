import { TestBed } from '@angular/core/testing';

import { MelanomPythonService } from './melanom-python.service';

describe('MelanomPythonService', () => {
  let service: MelanomPythonService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(MelanomPythonService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
