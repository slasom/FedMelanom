import { Component } from '@angular/core';
import { MelanomPythonService } from '../../services/melanom-python/melanom-python.service';
import { Patient } from '../../interfaces/patientInterface';
import * as sunData from '../../../assets/resources/sunExposure.json';
import { User } from '../../interfaces/userInterface';
import { SafeUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})
export class HomeComponent {

  patients: Patient[];
  labels: string[];
  selectedFile: File | null;
  imageSrc: SafeUrl;
  resultPrediction: string;
  sunExposureOptions: string[];
  zone: string;
  patientSelected: string;
  sunExposureSelected: string;
  showError: boolean;
  errorText: string;
  user: User;

  constructor(private melanomPythonService: MelanomPythonService){
    this.patients = [];
    this.labels  = [];
    this.selectedFile = null;
    this.imageSrc = '';
    this.resultPrediction = "";
    this.sunExposureOptions = this.getSunExposureOptions();
    this.zone = '';
    this.patientSelected = "";
    this.sunExposureSelected = "";
    this.showError = false;
    this.errorText = '¡ERROR!';
    
    let userStoraged = localStorage.getItem('user'),
        userObject;
    if(userStoraged) {
      userObject = JSON.parse(userStoraged);
    }
    this.user = {
      id: userObject?.id,
      name: userObject?.name
    }
  }

  ngOnInit(): void{
    this.setPatients();
  }

  /**
   * Update the image selected with file explorer
   * @param event Event emitted
   */
  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0] as File;
    if (this.selectedFile) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if(e.target?.result)
          this.imageSrc = e.target.result;
      };
      reader.readAsDataURL(this.selectedFile);
    }
  }

  /**
   * Predict if a image is a melanom or not
   */
  predictImage(): void {
    if(this.selectedFile != null) {
      console.log(this.zone, this.patientSelected, this.sunExposureSelected)
      if(this.zone != '' && this.patientSelected != '' && this.sunExposureSelected != '') {
        this.melanomPythonService.predictMelanom(this.user.id, this.selectedFile, this.patientSelected, this.sunExposureSelected, this.zone)
        .subscribe((response)=>{
            let prediction = response?.prediction;
            if(prediction) {
                this.resultPrediction = this.convertToPorcentage(prediction?.result);
            }
        }),
        (error: any) => {
            console.error('Error al realizar predicción', error);
            this.launchErrorMessage("Se ha producido un error al realizar la predicción");
        }
      }else {
        this.launchErrorMessage("Debes indicar el paciente, la zona afectada y la exposición solar");
      } 
    } else {
        this.launchErrorMessage("Primero debes seleccionar una imagen");
    }
  }

  /**
   * Converts prediction result in util porcentage value
   * @param number Prediction result value
   * @returns The value in porcentage
   */
  convertToPorcentage(number: number): string {
    const umbral = 0.0001;
    let resultadoPorcentaje: number;
    if (Math.abs(number) >= umbral) {
        resultadoPorcentaje = number * 100;
    } else {
        resultadoPorcentaje = 0;
    }
    return resultadoPorcentaje.toFixed(1) + '% de ser melanoma maligno';
  }

  /**
   * Update the patients for the dropdown
   */
  setPatients(): void {
    this.melanomPythonService.getPatients().subscribe(response => {
      this.patients = Object.values(response);
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
   * Display the error in the screen
   * @param error Error text message
   */
  launchErrorMessage(error: string) {
    this.showError = true;
    this.errorText = error;
    setTimeout(()=>{
      this.showError = false;
    }, 3000);
  }
}
