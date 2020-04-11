import React from "react";
import { Card, Container, CardHeader, Row, Col } from "reactstrap";

import CybneticsFooter from "components/CybneticsFooter.js";

class Login extends React.Component {
  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;
  }
  render() {
    return (
      <>
        <main ref="main">
          <section className="section">
          <Container className="pt-lg-7">
            <Row className="justify-content-center">
                <Col lg="8">
                  <Card className="bg-secondary shadow border-0">
                    <CardHeader className="bg-white pb-5">
                      <div className="text-muted text-center mb-3">
                        <h3>Login</h3>
                      </div>
                    </CardHeader>
                  </Card>
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

export default Login;
