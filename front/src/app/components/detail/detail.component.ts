import { Component, Input } from '@angular/core';
import { MelanomItemList } from '../../interfaces/melanomItemList';
import { ItemSelectedService } from '../../services/itemSelected/item-selected.service';
import { MelanomPythonService } from '../../services/melanom-python/melanom-python.service';
import { Observable } from 'rxjs';
import { SafeUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-detail',
  templateUrl: './detail.component.html',
  styleUrl: './detail.component.scss'
})
export class DetailComponent {

  @Input() data: MelanomItemList;

  defaultImage = '../../../assets/images/default.webp'

  originalImageUrl: SafeUrl = this.defaultImage;
  processedImageUrl: SafeUrl = this.defaultImage;

  isClicked: boolean = false;

  constructor(private itemSelectedService: ItemSelectedService, private melanomPythonService: MelanomPythonService){
    this.data = {
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
    this.itemSelectedService?.selectedItem$.subscribe(item => {
      this.data = item;
      this.setOriginalImage();
      this.setProcessedImage();
    });
  }

  /**
   * Function to obtain the observable of the inform image selected
   * @param imagePath Path of the image to load
   * @returns The observable function to obtain the image selcted
   */
  loadImage(imagePath: string): Observable<SafeUrl>{
    return this.melanomPythonService.getInformImage(imagePath);
  }

  /**
   * Set the value of the original image selected provided by the inform
   */
  setOriginalImage() {
    if(this.data && this.data.originalImage) {
      this.loadImage(this.data?.originalImage).subscribe(
        imageUrl => {
          this.originalImageUrl = imageUrl
        },
        error => {
          console.error('Error loading DETAIL original image:', error);
          this.originalImageUrl = this.defaultImage;
        }
      );
    }
  }

  /**
   * Set the value of the predicted image selected provided by the inform
   */
  setProcessedImage() {
    if(this.data && this.data.processedImage) {
      this.loadImage(this.data?.processedImage).subscribe(
        imageUrl => {
          this.processedImageUrl = imageUrl
        },
        error => {
          console.error('Error loading DETAIL predicted image:', error);
          this.processedImageUrl = this.defaultImage;
        }
      );
    }
  }

  changeImage() {
    this.isClicked = !this.isClicked;
  }
}
