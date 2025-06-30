import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { Observable, catchError, map, of, throwError } from 'rxjs';
import { User } from '../../interfaces/userInterface';

@Injectable({
  providedIn: 'root',
})
export class MelanomPythonService {
  private apiUrl: string = 'http://127.0.0.1:8000';
  
  defaultImage = '../../../assets/images/default.webp'

  constructor(private http: HttpClient, private sanitizer: DomSanitizer) {}

  /**
   * Check the login and get the information of the doctor
   * @param id --> Membership number
   * @param password --> Password asssigned to doctor
   * @returns Doctor information
   */
  login(id: string, password: string): Observable<any> {
    var loginData = {
      "id": id,
      "password": password
    };
    return this.http.post<any>(`${this.apiUrl}/login/`, loginData);
  }

  /**
   * Get all the patients of the database
   * @returns All patients of the database
   */
  getPatients(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/patients/`);
  }

  /**
   * Get all the informas that the doctor has generated
   * @param user -> Specific user
   * @returns All the informas that the doctor has generated
   */
  getInforms(user: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/informs/${user}`);
  }

  /**
   * Upload image of specific user
   * @param image -> Image to upload
   * @param user -> Specific user
   * @returns
   */
  uploadImage(image: File, user: string): Observable<any> {
    const formData = new FormData();
    formData.append('image', image);
    return this.http.post<any>(`${this.apiUrl}/upload_image/${user}`, formData);
  }

  /**
   * Retrain model of specific user
   * @param user -> Specific user
   * @returns
   */
  retrainModel(user: string) {
    const formData = new FormData();
    formData.append('user_id', user);
    return this.http.post<any>(`${this.apiUrl}/retrain_model/${user}`, formData);
  }

  /**
   * Check if the photo is melanom or not
   * @param user -> Specific user
   * @returns
   */
  predictSpecificImage(user: string, image: string) {
    const formData = new FormData();
    formData.append('user_id', user);
    formData.append('image', image);
    return this.http.post<any>(`${this.apiUrl}/predictSpecificImage/${user}/${image}`, formData);
  }

  /**
   * Check if the photo is melanom or not
   * @param user -> Specific user
   * @returns
   */
  predictMelanom(user: string, image: File, patient: string, sunExposure: string, zone: string) {
    const formData = new FormData();
    formData.append("image", image);
    return this.http.post<any>(`${this.apiUrl}/predictMelanom/${user}/${patient}/${zone}/${sunExposure}`, formData);
  }

  /**
   * Obtain labels of a specific file
   * @param user -> Specific user
   * @param filename -> Label.txt name
   * @returns
   */
  getLabels(user: string, filename: string) {
    return this.http.get<any>(`${this.apiUrl}/labels/${user}/${filename}`);
  }

  /**
   * Get a specific image of inform
   * @param imagePath Inform image provided
   * @returns Return image file
   */
  getInformImage(imagePath: string): Observable<SafeUrl> {
    return this.http.get(`${this.apiUrl}/${imagePath}`, { responseType: 'arraybuffer' }).pipe(
      map((imageArrayBuffer: ArrayBuffer) => {
        if (imageArrayBuffer.byteLength > 0) {
          const objectUrl = this.arrayBufferToBase64(imageArrayBuffer);
          return this.sanitizer.bypassSecurityTrustUrl(objectUrl);
        } else {
          return this.defaultImage;
        }
      }),
      catchError((error: HttpErrorResponse) => {
        console.error('Error fetching image:', error);
        return of(this.defaultImage);
      })
    );
  }

  /**
   * Do the average model with all the user models of the aplication
   * @param user -> Specific user
   * @returns 
   */
  federatedLearning(user: string) {
    const formData = new FormData();
    formData.append('user_id', user);
    return this.http.post<any>(`${this.apiUrl}/federatedLearning/${user}`, formData);
  }

  /**
   * Get the image in Base64 format
   * @param buffer Buffer of the image
   * @returns Image in Base64 format
   */
  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return 'data:image/jpeg;base64,' + window.btoa(binary);
  }

  /**
   * Save the settings of the specific user
   * @param user -> Specific user
   * @returns
   */
  saveSettingsUser(user: any): Observable<any> {
    return this.http.put(`${this.apiUrl}/users/settings/${user.id}`, user);
  }
}
