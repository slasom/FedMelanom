import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { MelanomPythonService } from '../../services/melanom-python/melanom-python.service';
import { UserService } from '../../services/user/user.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss'
})
export class LoginComponent {
  id: string = '';
  password: string = '';
  showError: boolean = false;
  errorText: string = '¡ERROR!';

  constructor(private Router: Router, private melanomPythonService: MelanomPythonService, private userService: UserService){}

  ngOnInit() {
    if(this.userService.getUser() != null){
      this.Router.navigate(['/home']);
    }
  }

  /**
   * Do the login with the credentials and get the doctor information
   */
  checkCredentials(){
    this.melanomPythonService.login(this.id, this.password)
      .subscribe((response)=>{
        if(response){
          this.userService.setUser(response);
          this.Router.navigate(['/home']);
        }
      },
      (error: any) => {
        console.error('Error al hacer login: ', error?.error?.detail);
        this.showError = true;
        this.errorText = error?.error?.detail;
        setTimeout(()=>{
          this.showError = false;
        }, 3000);
      }
    );
  }
}
