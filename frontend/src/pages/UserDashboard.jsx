import React from "react";
import { Card, Container, Form, Input, Button, Row, Col, CardHeader, CardBody, Modal, 
  ListGroup, Badge, Nav, NavItem, NavLink, FormGroup, Label, CustomInput} from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";

class UserDashboard extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      selectedFile: null,
      exampleModal: false,
    }
    this.handleComfirmation = this.handleComfirmation.bind(this);
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

  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;
  }
  render() {
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
                        <Col>
                        <Button className="btn-icon btn-3 float-right" color="primary" type="button" onClick={() => this.toggleModal("exampleModal")}>
                          Upload
                        </Button>
                        </Col>
                      </Row>
                        <Col className="ml-9">
                          <Modal
                            className="modal-dialog-centered"
                            isOpen={this.state.exampleModal}
                            toggle={() => this.toggleModal("exampleModal")}
                          >
                            <div className="modal-header">
                              <h5 className="modal-title" id="exampleModalLabel">
                                 Confirmation
                              </h5>
                              <button
                                aria-label="Close"
                                className="close"
                                data-dismiss="modal"
                                type="button"
                                onClick={() => this.toggleModal("exampleModal")}
                              >
                                <span aria-hidden={true}>×</span>
                              </button>
                            </div>
                            <div className="modal-body">
                              <Form>
                                <div className="mb-2">
                                  <Input required
                                  placeholder="Name" 
                                  id="name"
                                  type="text" 
                                  
                                  onChange={this.handleChange}
                                  />
                                </div>
                                <div className="mb-2">
                                  <Input required
                                  placeholder="Description" 
                                  id="description"
                                  type="text" 
                                  onChange={this.handleChange}
                                  />
                                </div>
                                <div className="mb-2">
                                  <Input required
                                  placeholder="Accuracy of your model" 
                                  id="acc"
                                  type="text" 
                                  onChange={this.handleChange}
                                  />
                                </div>
                              </Form>
                            </div>
                            <div className="modal-footer">
                              <Button className="btn-icon btn-3 ml-8" color="primary" type="button" onClick={this.handleComfirmation}>
                                Submit
                              </Button>
                            </div>
                          </Modal>
                        </Col>
                       
                    </Form>
                </CardHeader>
                <CardBody>
                  <div className="px-4">
                    <h2>
                      Uploaded Models
                    </h2>
                  </div>
                </CardBody>
                <CardBody>
                 <ListGroup>
                    
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


function MlModelTile(props){
  return (
    <div className="py-1">
    <Card className="shadow py-3">
      <Row className="align-items-center justify-content-md-between">
        <Col md="6">
          <span style={{"padding-left": 20}}>{props.datasetName}</span>
        </Col>
        <Col md="6">
          <Nav className="justify-content-end">
            <NavItem>
              <NavLink href="#" target="_blank" rel="noopener noreferrer">
                Details
              </NavLink>
            </NavItem>
            <NavItem>
              <NavLink href="#" target="_blank" rel="noopener noreferrer">
                <svg class="bi bi-pencil-square" width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                  <path d="M15.502 1.94a.5.5 0 010 .706L14.459 3.69l-2-2L13.502.646a.5.5 0 01.707 0l1.293 1.293zm-1.75 2.456l-2-2L4.939 9.21a.5.5 0 00-.121.196l-.805 2.414a.25.25 0 00.316.316l2.414-.805a.5.5 0 00.196-.12l6.813-6.814z"/>
                  <path fill-rule="evenodd" d="M1 13.5A1.5 1.5 0 002.5 15h11a1.5 1.5 0 001.5-1.5v-6a.5.5 0 00-1 0v6a.5.5 0 01-.5.5h-11a.5.5 0 01-.5-.5v-11a.5.5 0 01.5-.5H9a.5.5 0 000-1H2.5A1.5 1.5 0 001 2.5v11z" clip-rule="evenodd"/>
                </svg>
              </NavLink>
            </NavItem>
            <NavItem>
              <NavLink href="#" target="_blank" rel="noopener noreferrer">
                <svg class="bi bi-trash-fill" width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                  <path fill-rule="evenodd" d="M2.5 1a1 1 0 00-1 1v1a1 1 0 001 1H3v9a2 2 0 002 2h6a2 2 0 002-2V4h.5a1 1 0 001-1V2a1 1 0 00-1-1H10a1 1 0 00-1-1H7a1 1 0 00-1 1H2.5zm3 4a.5.5 0 01.5.5v7a.5.5 0 01-1 0v-7a.5.5 0 01.5-.5zM8 5a.5.5 0 01.5.5v7a.5.5 0 01-1 0v-7A.5.5 0 018 5zm3 .5a.5.5 0 00-1 0v7a.5.5 0 001 0v-7z" clip-rule="evenodd"/>
                </svg>
              </NavLink>
            </NavItem>
          </Nav>
        </Col>
      </Row>
    </Card>
    </div>
  );
}