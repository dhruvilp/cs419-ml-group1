import React from "react";
import {NavItem, NavLink, Nav, Container, Row, Col,} from "reactstrap";

class CybneticsFooter extends React.Component {
  render() {
    return (
      <>
        <footer className="has-cards">
          <Container>
            <Row className="align-items-center justify-content-md-between">
              <Col md="6">
                <div className="copyright">
                  © {new Date().getFullYear()}{" "}
                  <a href="https://github.com/dhruvilp/cs419-ml-group1" target="_blank" rel="noopener noreferrer">
                    Cybnetics.ml Team
                  </a>
                  .
                </div>
              </Col>
              <Col md="6">
                <Nav className="nav-footer justify-content-end">
                  <NavItem>
                    <NavLink href="https://github.com/dhruvilp/cs419-ml-group1" target="_blank" rel="noopener noreferrer">
                      About Us
                    </NavLink>
                  </NavItem>
                  <NavItem>
                    <NavLink href="https://github.com/dhruvilp/cs419-ml-group1" target="_blank" rel="noopener noreferrer">
                      MIT License
                    </NavLink>
                  </NavItem>
                </Nav>
              </Col>
            </Row>
          </Container>
        </footer>
      </>
    );
  }
}

export default CybneticsFooter;
