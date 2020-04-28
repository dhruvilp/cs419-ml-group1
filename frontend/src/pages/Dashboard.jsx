import React from "react";
import { Card, Container, Col, Row, Badge, ListGroup, Nav, NavItem, NavLink, CardBody } from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";
import UploadButtons from "components/UploadButtons";
import { userService } from "../services/user_service";

class Dashboard extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      ml_models: []
    }
  }

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
          ml_models : data
        });
      }
    })
    .catch((error) => {
      console.log(error);
    });
  }

  toggleModal = state => {
    this.setState({
      [state]: !this.state[state]
    });
  };

  render() {
    const { ml_models } = this.state;
    return (
      <>
        <CybneticsNavbar />
        <main ref="main">
          <section className="section-cybnetics-cover section-shaped my-0">
            <div className="shape shape-primary"></div>
            <div className="separator separator-bottom separator-skew">
              <svg xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none" version="1.1" viewBox="0 0 2560 100" x="0" y="0">
                <polygon className="fill-white" points="2560 0 2560 100 0 100"/>
              </svg>
            </div>
          </section>
          <section className="section">
            <Row className="py-3">
            <Col className="mt--200">
              <UploadButtons />
              <Container>
                <Card className="card-profile shadow mt--100">
                  <CardBody className="py-5">
                    <div>
                      <Col className="justify-content-center">
                        <div className="text-center">
                          <h3>Dashboard</h3>
                        </div>
                        <div style={{"paddingLeft": "inherit", "paddingRight": "inherit"}}>
                          <h5 className="text-left text-default font-weight-bold">Uploaded Datasets</h5>
                          <ListGroup>
                            <DatasetTile datasetName="1. MNIST" />
                            <DatasetTile datasetName="2. CIFAR-10" />
                          </ListGroup>
                        </div>
                        <div className="py-5" style={{"paddingLeft": "inherit", "paddingRight": "inherit"}}>
                          <h5 className="text-left text-default font-weight-bold">Uploaded ML Models</h5>
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
                        </div>
                      </Col>
                    </div>
                  </CardBody>
                </Card>
              </Container>
            </Col>
            </Row>
          </section>
        </main>
        <CybneticsFooter />
      </>
    );
  }

}

export default Dashboard;

function DatasetTile(props){
  return( 
    <div className="py-1"> 
    <Card className="shadow py-3">
      <Row className="align-items-center justify-content-md-between">
        <Col md="6">
          <span style={{"paddingLeft": 20}}>{props.datasetName}</span>
        </Col>
      </Row>
    </Card>
    </div>
  );
}
