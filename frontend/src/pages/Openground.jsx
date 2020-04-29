import React from "react";
import { Col, Card, Container, Button, FormGroup, Label, Input, Row, Progress} from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";

class Openground extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      file: null,
      fileName: '',
      attack: false,
    }
    this.loadResults = this.loadResults.bind(this);
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

  loadResults(){
    this.setState({attack: true})
  }

  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;
  }

  render() {

    if(this.state.attack === true){
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
            <section className="section">
              <Col className="mt--200">
              <Container>
                <Card className="card-profile shadow mt--200">
                  <div className="py-5">
                    <Col className="justify-content-center">
                      <div>
                      <FormGroup>
                        <Label for="exampleSelect">Select ML Model</Label>
                          <Input type="select" name="select" id="exampleSelect">
                            <option>1. MNIST (model tf-68)</option>
                            <option>2. CIFAR-10 (model tf-45)</option>
                          </Input>
                        </FormGroup>
                      </div>
                    </Col>
                  </div>
                  <Row className="mb-5 ml-3 mr-3">
                    <Col sm="6">
                      <Card className="shadow">
                        <img width="100%" src={this.state.file} alt={this.state.fileName} />
                      </Card>
                    </Col>
                    <Col sm="6">
                      <Card className="shadow">
                        <h4 className="ml-8">Predicted Results</h4>
                        <div className="progress-wrapper ml-3 mr-3">
                          <div className="progress-info">
                            <div className="progress-label">
                              <span>Predicted Label1</span>
                            </div>
                            <div className="progress-percentage">
                              <span>90%</span>
                            </div>
                          </div>
                          <Progress max="100" value="90" color="success" />
                        </div>
                        <div className="progress-wrapper ml-3 mr-3">
                          <div className="progress-info">
                            <div className="progress-label">
                              <span>Predicted Label2</span>
                            </div>
                            <div className="progress-percentage">
                              <span>12%</span>
                            </div>
                          </div>
                          <Progress max="100" value="12" color="success" />
                        </div>
                        <div className="progress-wrapper ml-3 mr-3">
                          <div className="progress-info">
                            <div className="progress-label">
                              <span>Predicted Label3</span>
                            </div>
                            <div className="progress-percentage">
                              <span>10%</span>
                            </div>
                          </div>
                          <Progress max="100" value="10" color="success" />
                        </div>
                        <Row className="ml-6 mb-3 mt-3">
                          <img alt="..." className="mb-3" src={require('assets/img/skull.PNG')}/>
                          <p className="ml-2 mt-1">Attacked Successfully!</p>
                          <img alt="..." className="mb-3 ml-3" src={require('assets/img/coin.png')}/>
                          <p className="ml-1 mt-1 text-primary">100pt</p>
                        </Row>
                      </Card>
                    </Col>
                  </Row>  
                </Card>
              </Container>
              </Col>
            </section>
          </main>
          <CybneticsFooter />
        </>
      );
    }
    
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
          <section className="section">
            <Col className="mt--200">
            <Container>
              <Card className="card-profile shadow mt--200">
                <div className="py-5">
                  <Col className="justify-content-center">
                    <div className="text-center">
                      <h2>Openground (Ethical Hacking)</h2>
                    </div>
                    <div>
                    <FormGroup>
                      <Label for="exampleSelect">Select ML Model</Label>
                        <Input type="select" name="select" id="exampleSelect">
                          <option>1. MNIST (model tf-68)</option>
                          <option>2. CIFAR-10 (model tf-45)</option>
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
                    <Button className="ml-9 mr-3" color="primary" type="button" size="lg" onClick={this.loadResults}>
                      Attack
                    </Button>
                    <i className="fa fa-chevron-right"></i>
                    <i className="fa fa-chevron-right"></i>
                    <i className="fa fa-chevron-right"></i>
                  </Col>
                </Row>
              </Card>
            </Container>
            </Col>
          </section>
        </main>
        <CybneticsFooter />
      </>
    );
  }
}

export default Openground;
