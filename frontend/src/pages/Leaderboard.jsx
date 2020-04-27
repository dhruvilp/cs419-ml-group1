import React from "react";
import { Badge, Card, Container, Row, Col, Table } from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";
import { userService } from "../services/user_service";

class Leaderboard extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      total_attempts: 0,
      total_successes: 0,
      gold_medals: 0,
      silver_medals: 0,
      bronze_medals: 0
    }
  }

  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;

    userService.getLeaderboard()
    .then((data) => {
      this.setState({
        total_attempts : data['total_attempts'],
        total_successes : data['total_successes'],
        gold_medals : data['gold_medals'],
        silver_medals : data['silver_medals'],
        bronze_medals : data['bronze_medals']
      });
      console.log(data);
    })
    .catch((error) => {
      console.log(error);
    });
  }
  render() {
    const {total_attempts, total_successes, gold_medals, silver_medals, bronze_medals} = this.state;
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
                          <span className="heading">{gold_medals}</span>
                          <span className="description">
                            <Badge color="gold" pill>
                              Gold $
                            </Badge>
                          </span>
                        </div>
                        <div>
                          <span className="heading">{silver_medals}</span>
                          <span className="description">
                            <Badge color="silver" pill>
                              Silver $
                            </Badge>
                          </span>
                        </div>
                        <div>
                          <span className="heading">{bronze_medals}</span>
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
                          <span className="heading">{total_attempts}</span>
                          <span className="description">Total Attempts</span>
                        </div>
                        <div>
                          <span className="heading">{total_successes}</span>
                          <span className="description">Total Successes</span>
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
                      <Table striped>
                        <thead>
                          <tr>
                            <th width="40">#</th>
                            <th style={{"text-align": "initial"}}>Username</th>
                            <th width="50">Total Points</th>
                            <th width="50">Total Attempts</th>
                            <th width="50">Total Sucesses</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <th scope="row">1</th>
                            <td style={{"text-align": "initial"}}>DeepCNet</td>
                            <td>1500</td>
                            <td>2</td>
                            <td>1</td>
                          </tr>
                          <tr>
                            <th scope="row">2</th>
                            <td style={{"text-align": "initial"}}>FrankSharp</td>
                            <td>800</td>
                            <td>2</td>
                            <td>2</td>
                          </tr>
                          <tr>
                            <th scope="row">3</th>
                            <td style={{"text-align": "initial"}}>MooseBot</td>
                            <td>300</td>
                            <td>2</td>
                            <td>1</td>
                          </tr>
                        </tbody>
                      </Table>
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
