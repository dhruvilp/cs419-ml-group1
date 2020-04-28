import React from "react";
import { Card, Container, Form, Input, Button, Row, Col, CardHeader, CardBody, Modal, 
  ListGroup, Nav, NavItem, NavLink, FormGroup, CustomInput, Badge} from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";
import { userService } from "../services/user_service";

class UserDashboard extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      selectedFile: null,
      confirmationModal: false,
      ml_models: []
    }
    this.handleComfirmation = this.handleComfirmation.bind(this);
  }

  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;

    userService.getListOfModels()
    .then((data) => {
      if(data){
        this.setState({
          ml_models : data
        });
      }
    })
    .catch((error) => {
      console.log(error);
    });

  }

  handleComfirmation(){
    console.log(document.getElementById("name").value)
  }

  toggleModal = state => {
    this.setState({
      [state]: !this.state[state]
    });
  };

  onChangeHandler=event=>{
    console.log(event.target.files[0])
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
            <Container>
              <Card className="card-profile shadow mt--300">
                <CardHeader className="card-header mt-2">
                  <Form role="form">
                    <Row>
                      <Col>
                        <FormGroup>
                          <CustomInput type="file" id="exampleCustomFileBrowser" name="file" onChange={this.onChangeHandler}/>
                        </FormGroup>
                      </Col>
                      <Col lg="2">
                        <Button className="btn-icon btn-3 float-right" color="primary" type="button" onClick={() => this.toggleModal("confirmationModal")}>
                          Upload
                        </Button>
                      </Col>
                    </Row>
                    <Col className="ml-9">
                      <Modal
                        className="modal-dialog-centered"
                        isOpen={this.state.confirmationModal}
                        toggle={() => this.toggleModal("confirmationModal")}
                      >
                        <div className="modal-header">
                          <h5 className="modal-title" id="confirmationModalLabel">
                              Confirmation
                          </h5>
                          <button
                            aria-label="Close"
                            className="close"
                            data-dismiss="modal"
                            type="button"
                            onClick={() => this.toggleModal("confirmationModal")}
                          >
                            <span aria-hidden={true}>×</span>
                          </button>
                        </div>
                        <div className="modal-body">
                          <Form>
                            <div className="mb-2">
                              <Input required
                              placeholder="Model Name" 
                              id="name"
                              type="text" 
                              
                              onChange={this.handleChange}
                              />
                            </div>
                            <div className="mb-2">
                              <Input required
                              placeholder="Model Description" 
                              id="description"
                              type="text" 
                              onChange={this.handleChange}
                              />
                            </div>
                            <div className="mb-2">
                              <Input required
                              placeholder="Accuracy of your model (Ex: .90)" 
                              id="acc"
                              type="number" 
                              onChange={this.handleChange}
                              />
                            </div>
                          </Form>
                        </div>
                        <div className="modal-footer">
                          <Button className="btn-icon btn-3 ml-8" color="primary" type="button" onClick={() => this.toggleModal("confirmationModal")}>
                            Submit
                          </Button>
                        </div>
                      </Modal>
                    </Col>
                  </Form>
                </CardHeader>
                <CardBody className="py-3">
                  <div>
                    <h3 className="py-3">Uploaded Models</h3>
                  </div>
                  <ListGroup>
                    {
                      ml_models.map((model, index) => 
                        <div className="py-1" key={model._id}>
                          <Card className="shadow py-3">
                            <Row className="align-items-center justify-content-md-between">
                              <Col md="8">
                                <span className="text-default" style={{"paddingLeft": 20, "fontSize": 20}}>{index+1}. {model.name}</span>
                                {model.attack_mode === 'black' ? <span style={{"paddingLeft": 10}}><Badge color="dark">Black Box</Badge></span> : <span></span>}
                                {model.attack_mode === 'white' ? <span style={{"paddingLeft": 10}}><Badge style={{backgroundColor: "#f2f2f2"}}>White Box</Badge></span> : <span></span>}
                                {model.attack_mode === 'gray' ? <span style={{"paddingLeft": 10}}><Badge style={{backgroundColor: "#787878", color: "#FFFFFF"}}>Gray Box</Badge></span> : <span></span>}
                              </Col>
                              <Col md="4">
                                <Nav className="justify-content-end">
                                  <NavItem>
                                    <NavLink href="#" target="_blank" rel="noopener noreferrer">
                                      <svg className="bi bi-trash-fill" width="2em" height="2em" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                                        <path fillRule="evenodd" d="M2.5 1a1 1 0 00-1 1v1a1 1 0 001 1H3v9a2 2 0 002 2h6a2 2 0 002-2V4h.5a1 1 0 001-1V2a1 1 0 00-1-1H10a1 1 0 00-1-1H7a1 1 0 00-1 1H2.5zm3 4a.5.5 0 01.5.5v7a.5.5 0 01-1 0v-7a.5.5 0 01.5-.5zM8 5a.5.5 0 01.5.5v7a.5.5 0 01-1 0v-7A.5.5 0 018 5zm3 .5a.5.5 0 00-1 0v7a.5.5 0 001 0v-7z" clipRule="evenodd"/>
                                      </svg>
                                    </NavLink>
                                  </NavItem>
                                </Nav>
                              </Col>
                            </Row>
                            <Row className="align-items-center justify-content-md-between">
                              <Col md="6">
                                <span style={{"paddingLeft": 20, "fontSize": 14}}>ID: {model._id}</span>
                              </Col>
                            </Row>
                            <Row className="align-items-center justify-content-md-between">
                              <Col>
                                <span style={{"paddingLeft": 20, "fontSize": 14}}>Description: {model.description}</span>
                              </Col>
                            </Row>
                          </Card>
                        </div>
                      )
                    }
                  </ListGroup>
                </CardBody>
              </Card>
            </Container>
          </section>
        </main>
        <CybneticsFooter />
      </>
    );
  }
}

export default UserDashboard;