import { Routes } from '@angular/router';
import { HomeComponent } from './containers/home/home.component';
import { LoginComponent } from './containers/login/login.component';
import { UserComponent } from './containers/user/user.component';
import { AppComponent } from './app.component';
import { InfoComponent } from './containers/info/info.component';
import { InformsComponent } from './containers/informs/informs.component';

export const routes: Routes = [
    {path: '', component: AppComponent},
    {path: 'home', component: HomeComponent},
    {path: 'login', component: LoginComponent},
    {path: 'account', component: UserComponent},
    {path: 'info', component: InfoComponent},
    {path: 'informs', component: InformsComponent}
];
