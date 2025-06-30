import { SafeUrl } from "@angular/platform-browser"

export interface MelanomItemList {
    id: string
    patient: string,
    idPatient: string,
    age: number,
    sex: string,
    zone: string,
    sunExposure: string,
    originalImage: string,
    processedImage: string,
    prediction: number
    date: string,
    selected: boolean
}