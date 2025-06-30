import { Component, Input, Output, EventEmitter } from '@angular/core';
import { Patient } from '../../interfaces/patientInterface';

@Component({
  selector: 'app-dropdown',
  templateUrl: './dropdown.component.html',
  styleUrl: './dropdown.component.scss'
})
export class DropdownComponent {

  @Input() type: string;
  @Input() patients: Patient[];
  @Input() items: string[];
  @Input() defaultText: string;
  @Input() extraClass: string;

  @Output() optionSelected = new EventEmitter<string>();

  selectedOption: number = 0;

  constructor() {
    this.type = 'string';
    this.patients = [];
    this.items = [];
    this.defaultText = "Seleccione una opción";
    this.extraClass = '';
  }

  /**
   * Handle the selection of a option in the dropdown
   * @param event event Emitter
   */
  selectOption(event: Event) {
    let optionSelected = (event.target as HTMLSelectElement).value;
    this.optionSelected.emit(optionSelected);
  }

  /**
   * Clena filter values options seleccted
   */
  cleanFilter() {
    this.selectedOption = 0;
    this.optionSelected.emit("");
  }

}
