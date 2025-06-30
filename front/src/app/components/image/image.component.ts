import { Component, ElementRef, Input, SimpleChanges, ViewChild } from '@angular/core';
import { SafeUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-image',
  templateUrl: './image.component.html',
  styleUrl: './image.component.scss'
})
export class ImageComponent {
  @Input() size: string;
  @Input() imageClass: string;
  @Input() src: SafeUrl;

  defaultImage = '../../../assets/images/default.webp';
  imageIdx: number = 0;

  constructor() {
    this.size = '10vh';
    this.imageClass = 'customImage';
    this.src = this.defaultImage;
  }

  ngOnChange(changes: SimpleChanges) {
    if(changes['src'] && changes['src'].currentValue) {
      this.src =  changes['src'].currentValue;
    }
  }
}
