import { Component } from '@angular/core';
import { College } from '../../interfaces/collegeInterface';
import { UserService } from '../../services/user/user.service';
import { MelanomPythonService } from '../../services/melanom-python/melanom-python.service';
@Component({
  selector: 'app-user',
  templateUrl: './user.component.html',
  styleUrl: './user.component.scss'
})
export class UserComponent {
  userInfo: College
  loading: boolean;
  showMessage: boolean;
  messageText: string;
  messageType: string;
  spinnerText: string;

  retrainText: string = "Realizando reentreno, por favor espere..."
  flText: string = "Realizando aprendizaje federado, por favor espere..."
  
  constructor(private userService: UserService,private melanomPythonService: MelanomPythonService) {
    this.userInfo = this.userService.getUser();
    this.loading = false;
    this.showMessage = false;
    this.messageText = '¡ERROR!';
    this.messageType = "error";
    this.spinnerText = this.retrainText;
  }

  /**
   * Do the retrain of the specific user model
   */
  retrainModel() {
    this.loading = true;
    this.spinnerText = this.retrainText;
    this.melanomPythonService.retrainModel(this.userInfo.id).subscribe(repsonse => {
      if(repsonse) {
        this.loading = false;
        this.messageType = "info";
        this.launchMessage("Reentreno realizado con éxito!")
      }
    }),
    (error: any) => {
        this.loading = false;
        console.error('Error al realizar el reentreno', error);
        this.messageType = "error";
        this.launchMessage("Se ha producido un error al realizar el reentreno");
    }
  }

  /**
   * Do the average model with all the user models of the aplication 
   */
  federatedLearning() {
    this.loading = true;
    this.spinnerText = this.flText;
    this.melanomPythonService.federatedLearning(this.userInfo.id).subscribe(
      response => {
        this.loading = false;
        this.messageType = "info";
        this.launchMessage("Aprendizaje federado realizado con éxito!");
      },
      (error: any) => {
          this.loading = false;
          console.error('Error al realizar el aprendizaje federado', error);
          this.messageType = "error";
          this.launchMessage("Se ha producido un error al realizar el aprendizaje federado");
      })
  }

  saveSettings(){
    console.log(this.userInfo)
    this.melanomPythonService.saveSettingsUser(this.userInfo).subscribe(repsonse => {
      if(repsonse) {
        this.userService.setUser(this.userInfo);
        this.messageType = "info";
        this.launchMessage("Ajustes guardados con éxito!")
      }
    }),
    (error: any) => {
        console.error('Error al guardar los ajustes', error);
        this.messageType = "error";
        this.launchMessage("Se ha producido un error al guardar los ajustes");
    }
  }

  /**
   * Display the message in the screen
   * @param message text message
   */
  launchMessage(message: string) {
    this.showMessage = true;
    this.messageText = message;
    setTimeout(()=>{
      this.showMessage = false;
    }, 3000);
  }

}
