import { Component, Input, SimpleChanges } from '@angular/core';

@Component({
  selector: 'app-message',
  templateUrl: './message.component.html',
  styleUrl: './message.component.scss'
})
export class MessageComponent {
  @Input() type: string = 'error'; // error, warning or info
  @Input() text: string = 'ERROR';
  @Input() show: boolean = false;

  color = this.getType(); // color of message container, default: red
  messageContainer = document.getElementById('containerM'); // message container

  constructor(){}

  /**
   * Show the message
   */
  ngOnChanges(changes: SimpleChanges) {
    if(changes['show'] && changes['show'].currentValue == true) {
      this.color = this.getType();
      this.showMessage();
    } else {
      this.hideMessage();
    }
  }

  showMessage() {
    this.messageContainer = document.getElementById('containerM');
    if(this.messageContainer) {
      this.messageContainer.style.opacity = '1';
    }
  }

  hideMessage() {
    this.messageContainer = document.getElementById('containerM');
    if(this.messageContainer) {
      this.messageContainer.style.opacity = '0';
    }
  }

  /**
   * Get the color of the popUp message (red,yellow or other)
   */
  getType() {
    let color = '#FF3040';
    if(this.type == 'warning') {
      color = '#FDFA4A';
    } else if(this.type == 'info'){
      color = '#5CD83B'
    }
    return color;
  }
}
