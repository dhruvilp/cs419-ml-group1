import React from "react";
import { Col, Card, Container, Button, CardBody, UncontrolledCollapse } from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";

class Openground extends React.Component {
  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;
  }

  toggleModal = state => {
    this.setState({
      [state]: !this.state[state]
    });
  };

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
            <Col className="mt--200">
            <Container>
              <Card className="card-profile shadow mt--200">
                <div className="py-5">
                  <Col className="justify-content-center">
                    <div className="text-center">
                      <h3>Openground (Ethical Hacking)</h3>
                    </div>
                    <div>
                      <Button block color="secondary" id="toggler" size="lg" type="button">
                        Search ML Models...
                      </Button>
                    </div>
                  </Col>
                </div>
                <UncontrolledCollapse toggler="#toggler">
                  <Card>
                    <CardBody>
                      Lorem ipsum dolor sit amet consectetur adipisicing elit. Nesciunt magni, voluptas debitis
                      similique porro a molestias consequuntur earum odio officiis natus, amet hic, iste sed
                      dignissimos esse fuga! Minus, alias.
                    </CardBody>
                  </Card>
                </UncontrolledCollapse>
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
