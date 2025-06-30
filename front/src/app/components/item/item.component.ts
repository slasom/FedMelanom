import { Component, Input, SimpleChanges } from '@angular/core';
import { MelanomItemList } from '../../interfaces/melanomItemList';
import { MelanomPythonService } from '../../services/melanom-python/melanom-python.service';
import { URL } from 'url';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-item',
  templateUrl: './item.component.html',
  styleUrl: './item.component.scss'
})
export class ItemComponent {
  @Input() item: MelanomItemList;

  defaultImage = '../../../assets/images/default.webp'

  originalImageUrl: SafeUrl = this.defaultImage;
  processedImageUrl: SafeUrl = this.defaultImage;


  constructor(private melanomPythonService: MelanomPythonService, private sanitizer: DomSanitizer) {
    this.item = {
      id: '',
      patient: '',
      idPatient: '',
      age: 0,
      sex: 'male',
      zone: '',
      sunExposure: '',
      originalImage: this.defaultImage,
      processedImage: this.defaultImage,
      prediction: 0,
      date: "",
      selected: false
    }
  }

  ngOnInit() {
    this.setOriginalImage();
    this.setProcessedImage();
  }

  /**
   * Function to obtain the observable of the inform image
   * @param imagePath Path of the image to load
   * @returns The observable function to obtain the image
   */
  loadImage(imagePath: string): Observable<SafeUrl>{
    return this.melanomPythonService.getInformImage(imagePath);
  }

  /**
   * Set the value of the original image provided by the inform
   */
  setOriginalImage() {
    console.log(this.item)
    if(this.item && this.item.originalImage) {
      this.loadImage(this.item.originalImage).subscribe(
        imageUrl => {
          this.originalImageUrl = imageUrl
        },
        error => {
          console.error('Error loading orifinal image:', error);
          this.originalImageUrl = this.defaultImage;
        }
      );
    }
  }

  /**
   * Set the value of the predicted image provided by the inform
   */
  setProcessedImage() {
    if(this.item && this.item.processedImage) {
      this.loadImage(this.item.processedImage).subscribe(
        imageUrl => {
          this.processedImageUrl = imageUrl
        },
        error => {
          console.error('Error loading predicted image:', error);
          this.processedImageUrl = this.defaultImage;
        }
      );
    }
  }
}
