import { Injectable } from '@angular/core';
import { College } from '../../interfaces/collegeInterface';

@Injectable({
  providedIn: 'root'
})
export class UserService {

  constructor() {}

  /**
   * Get the logged user
   * @returns The logged user
   */
  getUser() {
    let userStoraged = localStorage.getItem('user')
    if(userStoraged) {
      return JSON.parse(userStoraged);
    }
    return null
  }

  /**
   * Set the value in local storage provided by the login
   * @param user User provided in login screen
   */
  setUser(user: College) {
    localStorage.setItem("user", JSON.stringify(user));
  }

}
