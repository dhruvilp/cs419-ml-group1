import React from "react";
import { Badge, Card, Container, Row, Col } from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";

class Leaderboard extends React.Component {
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
                <div className="px-4">
                  <Row className="justify-content-center">
                    <Col className="order-lg-2" lg="3">
                      <div className="card-profile-image">
                        <img
                          alt="..."
                          className="rounded-circle"
                          src={require("assets/img/trophy.png")}
                        />
                      </div>
                    </Col>
                    <Col className="order-lg-2" lg="4">
                      <div className="card-profile-stats d-flex justify-content-center">
                        <div>
                          <span className="heading">25</span>
                          <span className="description">
                            <Badge color="gold" pill>
                              Gold $
                            </Badge>
                          </span>
                        </div>
                        <div>
                          <span className="heading">35</span>
                          <span className="description">
                            <Badge color="silver" pill>
                              Silver $
                            </Badge>
                          </span>
                        </div>
                        <div>
                          <span className="heading">43</span>
                          <span className="description">
                            <Badge color="bronze" pill>
                              Bronze $
                            </Badge>
                          </span>
                        </div>
                      </div>
                    </Col>
                    <Col className="order-lg-1" lg="4">
                      <div className="card-profile-stats d-flex justify-content-center">
                        <div>
                          <span className="heading">415</span>
                          <span className="description">Total Defenses</span>
                        </div>
                        <div>
                          <span className="heading">267</span>
                          <span className="description">Total Attacks</span>
                        </div>
                      </div>
                    </Col>
                  </Row>
                  <div className="text-center mt-5">
                    <h3>
                      Leaderboard
                    </h3>
                  </div>
                  <div className="text-center mt-5">
                    <Container>
                      <h6>
                        Leaderboard Table
                      </h6>
                    </Container>
                  </div>
                </div>
              </Card>
            </Container>
          </section>
        </main>
        <CybneticsFooter />
      </>
    );
  }
}

export default Leaderboard;
