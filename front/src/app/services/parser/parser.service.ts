import { Injectable } from '@angular/core';
import { MelanomItemList } from '../../interfaces/melanomItemList';

@Injectable({
  providedIn: 'root'
})
export class ParserService {

  constructor() { }

  /**
   * Parse the informs to follow the interface structure
   * @param informs Informs of a especific collegue
   * @returns The informs parsed with correct interface
   */
  parseinforms(informs: any[]): MelanomItemList[] {
      return informs.map(item => ({
        id: item?.id,
        patient: item?.Paciente,
        idPatient: item?.idPaciente,
        age: item?.Edad,
        sex: item?.Sexo,
        zone: item?.Zona,
        sunExposure: item?.ExposicionSolar,
        originalImage: item?.ImagenOriginal,
        processedImage: item?.ImagenProcesada,
        prediction: item?.Prediccion,
        date: item?.Fecha,
        selected: false
      }));
  }

}
