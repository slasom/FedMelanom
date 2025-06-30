import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NavbarComponent } from './components/navbar/navbar.component'
import { HomeComponent } from './containers/home/home.component';
import { LoginComponent } from './containers/login/login.component';
import { UserComponent } from './containers/user/user.component';
import { InfoComponent } from './containers/info/info.component';
import { ButtonComponent } from './components/button/button.component';
import { ImageComponent } from './components/image/image.component';
import { MessageComponent } from './components/message/message.component';
import { InformsComponent } from './containers/informs/informs.component';
import { ListComponent } from './components/list/list.component';
import { ItemComponent } from './components/item/item.component';
import { DetailComponent } from './components/detail/detail.component';
import { DropdownComponent } from './components/dropdown/dropdown.component';
import { FilterComponent } from './components/filter/filter.component';
import { LoadingComponent } from './components/loading/loading.component';

import { MelanomPythonService } from './services/melanom-python/melanom-python.service';
import { ItemSelectedService } from './services/itemSelected/item-selected.service';
import { UserService } from './services/user/user.service';

import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

@NgModule({
  declarations: [
    AppComponent,
    NavbarComponent,
    ButtonComponent,
    ImageComponent,
    HomeComponent,
    LoginComponent,
    UserComponent,
    InfoComponent,
    MessageComponent,
    InformsComponent,
    ListComponent,
    ItemComponent,
    DetailComponent,
    DropdownComponent,
    FilterComponent,
    LoadingComponent
  ],
  providers: [
    MelanomPythonService,
    ItemSelectedService,
    UserService
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule
  ],
  bootstrap: [AppComponent],
})
export class AppModule { }
