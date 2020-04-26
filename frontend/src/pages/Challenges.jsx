import React from "react";
import { Card, Container, Row, Col, Button, CardBody, Nav, NavItem, NavLink, TabContent, TabPane} from "reactstrap";
import classnames from "classnames";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";
//import Test from "pages/test.txt"

class Challenges extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      tabs: 1,
      user: {}
    };
    
    this.downloadDatasets = this.downloadDatasets.bind(this);
    this.downloadModels = this.downloadModels.bind(this);
    this.toggleNavs = this.toggleNavs.bind(this);
  }

  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;
    this.setState({
      user: JSON.parse(localStorage.getItem('user'))
    });
  }

  toggleNavs = (e, state, index) => {
    e.preventDefault();
    this.setState({
      [state]: index
    });
  };

  downloadDatasets(){
    console.log("here");
    var file = 'text.txt'
    var blob = new Blob([ file ], {
      type : "text"
    });
    var link = document.createElement("a");
    var t = URL.createObjectURL(blob);
    link.href = t;
    link.style = "visibility:hidden";
    link.download = 'text.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  downloadModels(){
    /*
    var file_path = 'pages/test.txt';
    var a = document.createElement('A');
    a.href = file_path;
    a.download = file_path;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);*/
  }

  render() {
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
            <Container className="mt--200">
              <Row>
                <Col>
                  <div className="text-start mt--200">
                    <p className="text-white">
                      <span className="h1 text-white"><strong>Cybnetics ML</strong></span>
                      <span className="h2 text-white"> is a platform for students to perform adversial attack 
                      on trained machine learning models for learning purposes</span>
                    </p>
                  </div>
                  <div className="text-start mt-7">
                    <p>
                      <span className="h3">Users can publish their trained models on MNIST &
                       CIFAR-10 and perform attacks on other user's models to get points</span>
                    </p>
                  </div>
                </Col>
                <Col>
                  <div className="mt--9">
                    <Nav
                      className="nav-fill flex-column flex-md-row"
                      id="tabs-icons-text"
                      pills
                      role="tablist"
                    >
                      <NavItem>
                        <NavLink
                          aria-selected={this.state.tabs === 1}
                          className={classnames("mb-sm-3 mb-md-0", {
                          active: this.state.tabs === 1
                          })}
                          onClick={e => this.toggleNavs(e, "tabs", 1)}
                          href="#pablo"
                          role="tab"
                        >
                        Download Datasets
                      </NavLink>
                    </NavItem>
                    <NavItem>
                        <NavLink
                          aria-selected={this.state.tabs === 2}
                          className={classnames("mb-sm-3 mb-md-0", {
                          active: this.state.tabs === 2
                          })}
                          onClick={e => this.toggleNavs(e, "tabs", 2)}
                          href="#pablo"
                          role="tab"
                        >
                        Download ML Models
                      </NavLink>
                    </NavItem>
                    </Nav>
                  </div>

                  {/*========================================
                                  DATASET TAB 
                   =========================================*/}
                  <div className="mt-5">
                    <Card className="bg-secondary shadow border-0">
                      <CardBody >
                        <TabContent activeTab={"tabs" + this.state.tabs}>
                          <TabPane tabId="tabs1">
                          <div>
                          <h1>
                            MNIST
                            <Button className="btn-icon btn-3 ml-8" color="primary" type="button" onClick={this.downloadDatasets}>
                              <span className="btn-inner--icon">
                                <i className="fa fa-download"></i>
                              </span>
                              <span className="btn-inner--text">Download</span>
                            </Button>
                          </h1>
                          <h5>
                          The MNIST database of handwritten digits,
                          available from this page, has a training set of 60,000 examples…
                          </h5>
                        </div>
                        <div>
                          <h1>
                            CIFAR-10
                            <Button className="btn-icon btn-3 ml-7" color="primary" type="button">
                              <span className="btn-inner--icon">
                                <i className="fa fa-download"></i>
                              </span>
                              <span className="btn-inner--text">Download</span>
                            </Button>
                          </h1>
                          <h5>
                          The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
                          with 6000 images per class. There are 50000 training images….
                          </h5>
                        </div>
                          </TabPane>
                        </TabContent>

                        {/*========================================
                                        ML MODEL TAB 
                         =========================================*/}
                        <TabContent activeTab={"tabs" + this.state.tabs}>
                          <TabPane tabId="tabs2">
                            <p>
                              Download ML Models...
                            </p>
                          </TabPane>
                        </TabContent>
                      </CardBody> 
                    </Card>
                  </div>
                        
                      
                </Col>
              </Row>
            </Container>
          </section>
        </main>
        <CybneticsFooter />
      </>
    );
  }
}

export default Challenges;
