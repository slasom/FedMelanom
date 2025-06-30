import { Component, Input, SimpleChange, SimpleChanges } from '@angular/core';
import { ItemSelectedService } from '../../services/itemSelected/item-selected.service';
import { MelanomItemList } from '../../interfaces/melanomItemList';

@Component({
  selector: 'app-list',
  templateUrl: './list.component.html',
  styleUrl: './list.component.scss'
})
export class ListComponent {
  @Input()items: MelanomItemList[];
  @Input()filter: { patient: string, sunExposure: string, date: string };

  allItems: MelanomItemList[];

  constructor(private itemSelectedService: ItemSelectedService){
    this.items = [];
    this.filter = {
      patient: '',
      sunExposure: '',
      date: ''
    }
    this.allItems = [];
  }

  ngOnChanges(changes: SimpleChanges) {
    if(changes['items'] && changes['items'].currentValue && changes['items'].currentValue.length > 0) {
      this.items =  changes['items'].currentValue;
      this.items[0].selected = true;
      this.itemSelectedService.setSelectedItem(this.items[0]); 
      this.allItems = this.items;
    }
    if(changes['filter'] && changes['filter'].currentValue) {
      this.filter =  changes['filter'].currentValue;
      this.items = this.filterSelected(this.filter?.patient, this.filter?.sunExposure, this.filter?.date);
      setTimeout(()=>{
        this.selectItem(this.items[0].id); 
      }, 100);
    }
  }

  /**
   * Change the ithem selected and the detail item
   * @param id Id of the item list selected
   */
  selectItem(id: string) {
    this.items.forEach((item)=>{
      if(item.id == id) {
        item.selected = true;
        this.itemSelectedService.setSelectedItem(item);
      }else{
        item.selected = false;
      }
    });
  }

  /**
   * Get the items that match with the filter options
   * @param patient patient selected in filter
   * @param sunExposure sun exposure selected in filter
   * @param date date selected in filter
   */
  filterSelected(patient: string, sunExposure: string, date: string) {
    return this.allItems.filter(item => {
      return (
        ((patient && patient != '') ? item.idPatient == patient : true) &&
        ((sunExposure && sunExposure != '') ? item.sunExposure == sunExposure : true) &&
        ((date && date != '') ? item.date == date : true)
      )
    });
  }
}
