import { Component } from '@angular/core';
import { College } from '../../interfaces/collegeInterface';
import { UserService } from '../../services/user/user.service';
import { MelanomPythonService } from '../../services/melanom-python/melanom-python.service';
import { MelanomItemList } from '../../interfaces/melanomItemList';
import { ParserService } from '../../services/parser/parser.service';

@Component({
  selector: 'app-informs',
  templateUrl: './informs.component.html',
  styleUrl: './informs.component.scss'
})
export class InformsComponent {
  user: College;
  informs: MelanomItemList[]
  filter = { patient: '', sunExposure: '', date: '' };

  constructor(private userService: UserService, private melanomPythonService: MelanomPythonService, private parserService: ParserService) {
    this.user = this.userService.getUser();
    this.informs = [];
  }

  ngOnInit() {
    this.setInforms();
  }

  /**
   * Set the informs of the college
   */
  setInforms() {
    this.melanomPythonService.getInforms(this.user.id).subscribe(response => {
      this.informs = this.parserService.parseinforms(response?.informs);
    });
  }

  /**
   * Event receiver when the filter is clicked
   * @param newFilter Filter options selected
   */
  onFilterSelected(newFilter: { patient: string, sunExposure: string, date: string }) {
    this.filter = newFilter;
  }

}
