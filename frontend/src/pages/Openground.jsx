import React from "react";
import { Col, Card, Container, Button, FormGroup, Label, Input, Row} from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";
import { userService } from "../services/user_service";

class Openground extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      file: null,
      fileName: '',
      ml_models: [],
      'selectModel': ''
    }
    this.handleChange = this.handleChange.bind(this);
    this.dropHandler = this.dropHandler.bind(this);
    this.dragOverHandler = this.dragOverHandler.bind(this);
  }

  dropHandler=ev=>{
    console.log('File(s) dropped');
  
    // Prevent default behavior (Prevent file from being opened)
    ev.preventDefault();

    if (ev.dataTransfer.items[0].kind === 'file') {
      var f = ev.dataTransfer.items[0].getAsFile();
      this.setState({
        file: URL.createObjectURL(f),
        fileName: f.name
      })
    }
  }

  dragOverHandler=ev=>{
    console.log('File(s) in drop zone'); 
  
    // Prevent default behavior (Prevent file from being opened)
    ev.preventDefault();
  }

  toggleModal = state => {
    this.setState({
      [state]: !this.state[state]
    });
  };

  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;

    // var jwt = require('jsonwebtoken');
    // var username = jwt.decode(JSON.parse(localStorage.getItem('user'))['token'])['username'];

    userService.getListOfModels()
    .then((data) => {
      if(data){
        this.setState({
          ml_models: data
        });
      }
    })
    .catch((error) => {
      console.log(error);
    });
  }

  handleChange(e) {
    const { name, value } = e.target;
    this.setState({ [name]: value });
  }

  render() {
    const { ml_models } = this.state;
    return (
    <>
      <CybneticsNavbar />
      <main className="profile-page" ref="main">
        <section className="section-cybnetics-cover section-shaped my-0">
          <div className="shape shape-primary"></div>
          <div className="separator separator-bottom separator-skew">
            <svg xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none" version="1.1" viewBox="0 0 2560 100" x="0" y="0">
              <polygon className="fill-white" points="2560 0 2560 100 0 100"/>
            </svg>
          </div>
        </section>
        <section className="section mt--100">
          {!this.state.attackModal 
            ? <Container>
            <Card className="card-profile shadow mt--300">
              <div className="py-5">
                <Col className="justify-content-center">
                  <div className="text-center">
                    <h2>Openground (Ethical Hacking)</h2>
                  </div>
                  <div>
                  <FormGroup>
                    <Label for="selectModel">ML Models <span style={{color: '#FF0000'}}>*</span></Label>
                    <Input type="select" name="selectModel" id="selectModel" onChange={this.handleChange}>
                      <option value="">Select model...</option>
                      {ml_models.map((model, index) => 
                      <option key={model._id} value={model._id}>
                          {index+1}. {model.name} ({model._id})
                      </option>
                      )}
                    </Input>
                  </FormGroup>
                  </div>
                </Col>
              </div>                
              <div className="drag-drop-zone" onDrop={this.dropHandler} onDragOver={this.dragOverHandler}>
              <div className="ml-9 mb-5" style={{
                  border: 'dashed lightgrey 3px', width: 700, height: 300}}>
                  <Col className="ml-9">
                    <div className="ml-6 mb-5">
                      <h2>Upload File</h2>
                    </div>
                    <div className="ml-8 mb-5">
                      <i className="fa fa-upload fa-4x"></i>
                    </div>
                    <div className="ml-6">
                      <p>Drag and drop a file here</p>
                    </div>
                  </Col>
                </div>
              </div>
              <Row className="ml-7 mb-4">
                <Col>
                  <h6>Selected file: {this.state.fileName}</h6>
                </Col>
                <Col>
                  <Button className="ml-9 mr-3" color="primary" type="button" size="lg" onClick={() => this.toggleModal('attackModal')}>
                    Attack
                  </Button>
                </Col>
              </Row>
            </Card>
          </Container>
          : <Container>
          <Card className="card-profile shadow mt--300">
            <div className="py-5">
              <Col className="justify-content-center">
                <div>
                  <Button block color="primary" type="button" size="lg" onClick={() => this.toggleModal('attackModal')}>
                    Attack Again
                  </Button>
                </div>
              </Col>
            </div>
            <Row className="mb-5 ml-3 mr-3">
              <Col sm="6">
                <Card className="shadow">
                  <img width="100%" src={this.state.file} alt={this.state.fileName} />
                </Card>
              </Col>
              <Col sm="6" className="align-items-center justify-content-md-between">
                <Card className="shadow">
                  <h4 className="text-center py-3">Results</h4>
                  <Col>
                    <Row className="align-items-center justify-content-md-between">
                      <Col sm="2">
                      <img alt="..." className="mb-3 text-center" src={require('assets/img/skull.PNG')}/>
                      </Col>
                      <Col>
                      <p className="text-left">Attacked Successfully!</p>
                      </Col>
                    </Row>
                    <Row className="align-items-center justify-content-md-between">
                      <Col sm="2">
                      <img alt="..." className="mb-3 text-center" src={require('assets/img/coin.png')}/>
                      </Col>
                      <Col>
                      <p className="text-primary text-left">1 pt</p>
                      </Col>
                    </Row>
                  </Col>
                </Card>
              </Col>
            </Row>  
          </Card>
        </Container>
          }
        </section>
      </main>
      <CybneticsFooter />
    </>
    );

  }
}

export default Openground;
