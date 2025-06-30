import { Component } from '@angular/core';
import * as text from '../../../assets/resources/info.json';

@Component({
  selector: 'app-info',
  templateUrl: './info.component.html',
  styleUrl: './info.component.scss'
})
export class InfoComponent {

  content: string;

  constructor(){
    this.content = "";
  } 

  ngOnInit(){
    this.content = text.text[0];
  }
}
