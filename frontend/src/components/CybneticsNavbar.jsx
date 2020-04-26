import React from "react";
import { Link } from "react-router-dom";
import Headroom from "headroom.js";
import {Button, UncontrolledCollapse, NavbarBrand, Navbar, NavItem, NavLink, Nav, Container, Row, Col,} from "reactstrap";

class CybneticsNavbar extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      collapseClasses: "",
      collapseOpen: false,
      isAdmin: false
    };
  }

  componentDidMount() {
    let headroom = new Headroom(document.getElementById("navbar-main"));
    headroom.init();

    var jwt = require('jsonwebtoken');
    var decoded = jwt.decode(JSON.parse(localStorage.getItem('user'))['token']);
    this.setState({ 
      isAdmin: decoded['admin']
    });
  }

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
    const { isAdmin, collapseClasses } = this.state;
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
                className={collapseClasses}
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
                  {isAdmin &&
                    <NavItem>
                      <NavLink href="/dashboard">
                        Dashboard
                      </NavLink>
                    </NavItem>
                  }
                  {!isAdmin &&
                    <NavItem>
                      <NavLink href="/challenges">
                        Challenges
                      </NavLink>
                    </NavItem>
                  }
                  {!isAdmin &&
                    <NavItem>
                      <NavLink href="/userDashboard">
                        Dashboard
                      </NavLink>
                    </NavItem>
                  }
                  <NavItem>
                    <NavLink href="/leaderboard">
                      Leaderboard
                    </NavLink>
                  </NavItem>
                  {!isAdmin &&
                    <NavItem>
                      <NavLink href="/openground">
                        Openground
                      </NavLink>
                    </NavItem>
                  }
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
