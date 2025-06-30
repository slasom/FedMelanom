import { Component, Output, EventEmitter } from '@angular/core';
import { Patient } from '../../interfaces/patientInterface';
import { MelanomPythonService } from '../../services/melanom-python/melanom-python.service';
import * as sunData from '../../../assets/resources/sunExposure.json';
@Component({
  selector: 'app-filter',
  templateUrl: './filter.component.html',
  styleUrl: './filter.component.scss'
})
export class FilterComponent {

  patients: Patient[];
  sunExposureOptions: string[];
  isVisible: boolean;
  patientSelected: string;
  sunExposureSelected: string;
  dateSelected: string;

  @Output() filterSelected = new EventEmitter<{patient: string, sunExposure: string, date: string}>();

  constructor(private melanomPythonService: MelanomPythonService){
    this.patients = [];
    this.sunExposureOptions = [];
    this.isVisible = false;
    this.patientSelected = '';
    this.sunExposureSelected = '';
    this.dateSelected = '';
  }

  ngOnInit() {
    this.setPatientsOptions();
    this.sunExposureOptions = this.getSunExposureOptions();
  }

  /**
   * Set the value for the patients options
   */
  setPatientsOptions() {
    this.melanomPythonService.getPatients().subscribe(patients => {
      this.patients = Object.values(patients);
    });
  }

  /**
   * Get the info for the sun exposure options dropdown
   * @returns Array of string with the sun exposure options
   */
  getSunExposureOptions(): string[] {
    let options: string[] = [];
    sunData.sunExposure.forEach(item => {
      options.push(item.label);
    });
    return options;
  }

  /**
   * Show the filter if clicks in the arrow
   */
  showFilter() {
    this.isVisible = !this.isVisible;
  }

  /**
   * Handle patient option selected
   * @param selectedPatient option selected
   */
  handlePatientSelected(selectedPatient: string) {
    this.patientSelected = selectedPatient;
  }

  /**
   * Handle sun exposure option selected
   * @param sunExposureSelected option selected
   */
  handleSunExposureSelected(selectedSunExposure: string) {
    this.sunExposureSelected = selectedSunExposure;
  }

  /**
   * Get the date with the correct format
   * @param date Date selected in input
   * @returns The date with the correct format
   */
  formatDate(date: string): string {
    console.log(date)
    if(date){
      const [year, month, day] = date.split('-');
      return `${day}/${month}/${year}`;
    }else{
      return '';
    }
  }

  /**
   * Send event with the filter options selected
   */
  searchResults() {
    this.filterSelected.emit({
      patient: this.patientSelected,
      sunExposure: this.sunExposureSelected,
      date: this.formatDate(this.dateSelected)
    });
  }
}
