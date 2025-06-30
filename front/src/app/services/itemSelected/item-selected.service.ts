import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ItemSelectedService {

  private selectedItemSource = new BehaviorSubject<any>(null);
  selectedItem$ = this.selectedItemSource.asObservable();

  constructor() { }

  setSelectedItem(item: any) {
    this.selectedItemSource.next(item);
  }
}
