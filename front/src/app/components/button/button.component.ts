import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-button',
  templateUrl: './button.component.html',
  styleUrl: './button.component.scss'
})
export class ButtonComponent {
  @Input() size: string = '1.5vh';
  @Input() color: string = '#b1afff';
  @Input() text: string = ''
  @Input() buttonClass: string = '';
}
