import React from "react";
import { Link } from "react-router-dom";
import Headroom from "headroom.js";
import {Button, UncontrolledCollapse, NavbarBrand, Navbar, NavItem, NavLink, Nav, Container, Row, Col,} from "reactstrap";

class CybneticsNavbar extends React.Component {
  componentDidMount() {
    let headroom = new Headroom(document.getElementById("navbar-main"));
    headroom.init();
  }
  state = {
    collapseClasses: "",
    collapseOpen: false
  };

  onExiting = () => {
    this.setState({
      collapseClasses: "collapsing-out"
    });
  };

  onExited = () => {
    this.setState({
      collapseClasses: ""
    });
  };

  render() {
    return (
      <>
        <header className="header-global">
          <Navbar
            className="navbar-main navbar-transparent navbar-light headroom"
            expand="lg"
            id="navbar-main"
          >
            <Container>
              <NavbarBrand className="mr-lg-5" to="/" tag={Link}>
                <img
                  alt="..."
                  src={require("assets/img/logo_text_white.png")}
                />
              </NavbarBrand>
              <button className="navbar-toggler" id="navbar_global">
                <span className="navbar-toggler-icon" />
              </button>
              <UncontrolledCollapse
                toggler="#navbar_global"
                navbar
                className={this.state.collapseClasses}
                onExiting={this.onExiting}
                onExited={this.onExited}
              >
                <div className="navbar-collapse-header">
                  <Row>
                    <Col className="collapse-brand" xs="6">
                      <Link to="/">
                        <img
                          alt="..."
                          src={require("assets/img/logo_text_cyan.png")}
                        />
                      </Link>
                    </Col>
                    <Col className="collapse-close" xs="6">
                      <button className="navbar-toggler" id="navbar_global">
                        <span />
                        <span />
                      </button>
                    </Col>
                  </Row>
                </div>
                <Nav className="align-items-lg-center ml-lg-auto" navbar>
                  {/* Make DASHBOARD only available to ADMIN */}
                  <NavItem>
                    <NavLink href="/dashboard">
                      Dashboard
                    </NavLink>
                  </NavItem>
                  {/* Make CHALLENGES only available to USER */}
                  <NavItem>
                    <NavLink href="/challenges">
                      Challenges
                    </NavLink>
                  </NavItem>
                  <NavItem>
                    <NavLink href="/leaderboard">
                      Leaderboard
                    </NavLink>
                  </NavItem>
                  {/* Make OPENGROUND only available to USER */}
                  <NavItem>
                    <NavLink href="/openground">
                      Openground
                    </NavLink>
                  </NavItem>
                  <NavItem className="d-none d-lg-block ml-lg-4">
                    <Button className="btn-neutral btn-icon" color="default" href="/login">
                      Logout
                    </Button>
                  </NavItem>
                </Nav>
              </UncontrolledCollapse>
            </Container>
          </Navbar>
        </header>
      </>
    );
  }
}

export default CybneticsNavbar;
