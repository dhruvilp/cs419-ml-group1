import React from "react";
import { Card, Container, Row, Col, Button, CardBody} from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar.js";
import CybneticsFooter from "components/CybneticsFooter.js";
import FontAwesomeIcon from "assets/font-awesome/css/font-awesome.min.css"

class Challenges extends React.Component {
  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;
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
                </Col>
                <Col>
                  <div className="mt--9">
                    <Row>
                      <Col>
                      <Button color="success" type="button" size="lg">Download Datasets</Button>
                      </Col>
                      <Col>
                      <Button color="secondary" type="button" size="lg">Download ML Models</Button>
                      </Col>
                    </Row>
                  </div>
                  <div className="mt-5">
                    <Card className="bg-secondary shadow border-0">
                      <CardBody >
                        <div>
                          <h1>
                            MNIST
                            <Button className="btn-icon btn-3 ml-2" color="primary" type="button">
                              <span className="btn-inner--icon">
                                <FontAwesomeIcon icon={downloadfa-download}></FontAwesomeIcon>
                              </span>
                              <span className="btn-inner--text">With icon</span>
                            </Button>
                          </h1>
                         
                          
                        </div>
                      </CardBody> 
                    </Card>
                  </div>
                </Col>
              </Row>
              <Row className="mt-8">
                <Col>
                  <div className="text-start mt-4">
                    <p>
                      <span className="h3">Users can publish their trained models on MNIST &
                       CIFAR-10 and perform attacks on other user's models to get points</span>
                    </p>
                  </div>
                </Col>
                <Col>
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
