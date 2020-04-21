import React from "react";
import { Card, Container, Row, Col, Button, CardBody} from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";

class Challenges extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      username: ''
      // user: {}
    };
  }

  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;

    this.setState({
      username: localStorage.getItem('user')
    });
    // this.setState({
    //   user: JSON.parse(localStorage.getItem('user'))
    // });
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
                            <Button className="btn-icon btn-3 ml-8" color="primary" type="button">
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
