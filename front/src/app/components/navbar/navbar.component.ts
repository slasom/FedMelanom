import { Component } from '@angular/core';
import * as navBarItemsData from '../../../assets/resources/navBarItems.json'
import { Router } from '@angular/router';

@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.scss'
})
export class NavbarComponent {
  navBarItems: {label: string, link: string}[] = navBarItemsData.navbarItemsUnlogged;

  constructor(private Router: Router){}

  ngOnInit() {
    let userItem = localStorage.getItem("user");
    if(userItem) {
      this.navBarItems = navBarItemsData.navbarItemsLogged;
    }else{
      this.navBarItems = navBarItemsData.navbarItemsUnlogged;
    }
  }

  /**
   * Check the link and redirect to this page
   * @param link Page to redirect
   */
  redirectTo(link: string){
    if(link == '/logout') {
      localStorage.removeItem("user");
      this.Router.navigate(['/login']);
    }else{
      this.Router.navigate([link]);
    }
  }
  
}
