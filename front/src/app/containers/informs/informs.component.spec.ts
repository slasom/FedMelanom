import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InformsComponent } from './informs.component';

describe('InformsComponent', () => {
  let component: InformsComponent;
  let fixture: ComponentFixture<InformsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InformsComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(InformsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
