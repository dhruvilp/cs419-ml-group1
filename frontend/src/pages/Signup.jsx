import React from "react";
import { Button, Card, CardImg, CardHeader, CardBody, Container, Form, FormFeedback, FormGroup, Input, InputGroup, InputGroupAddon, InputGroupText, Row, Col, Spinner, UncontrolledAlert } from "reactstrap";

import { authService } from '../services/auth_service';
import CybneticsFooter from "components/CybneticsFooter";

class Signup extends React.Component {

  constructor(props) {
    super(props);

    authService.logout();

    this.state = {
      'username': '',
      'password': '',
      submitted: false,
      loading: false,
      error: false
    }

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;
  }

  handleChange(e) {
    const { name, value } = e.target;
    this.setState({ [name]: value });
  }

  handleSubmit(e) {
    e.preventDefault();

    this.setState({ submitted: true });
    const { username, password } = this.state;

    if (!(username && password)) {
        return;
    }

    this.setState({ loading: true });
    authService.signup(username, password)
      .then(
        user => {
          const { from } = this.props.location.state || { from: { pathname: "/challenges" } };
          this.props.history.push(from);
        },
        error => {
          console.log(error);
          this.setState({ 
            error: true, 
            loading: false
          });
          return;
        }
      );
  }

  render() {
    const { username, password, submitted, loading, error } = this.state;
    return (
      <>
        <main ref="main">
          <section className="section section-md pt-0">
          <Container className="pt-lg-5">
              <Row className="justify-content-center">
                <Col lg="6">
                  <Card className="bg-secondary shadow border-0">
                    <CardHeader className="bg-primary pb-3">
                      <CardImg
                        alt="..."
                        src={require("assets/img/logo_text_white.png")}
                        top
                      />
                    </CardHeader>
                    <CardBody className="bg-white px-lg-5 py-lg-5">
                      <div className="text-muted text-center mb-3">
                        <h3 className="text-primary">
                          Signup
                        </h3>
                      </div>
                      { error ?
                        <UncontrolledAlert color="danger">
                          Error occurred, try again!
                        </UncontrolledAlert>
                        : <span></span>
                      }
                      <Form role="form" onSubmit={this.handleSubmit}>
                        <FormGroup className="mb-3">
                          <InputGroup className="input-group-alternative">
                            <InputGroupAddon addonType="prepend">
                              <InputGroupText>
                                <svg className="bi bi-person-fill" width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                                  <path fillRule="evenodd" d="M3 14s-1 0-1-1 1-4 6-4 6 3 6 4-1 1-1 1H3zm5-6a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd"/>
                                </svg>
                              </InputGroupText>
                            </InputGroupAddon>
                            <Input required
                              placeholder="Username" 
                              type="email" 
                              id="username" 
                              name="username"
                              value={username}
                              onChange={this.handleChange}
                              invalid = { submitted && !username }
                            />
                            <FormFeedback>username is required!</FormFeedback>
                          </InputGroup>
                        </FormGroup>
                        <FormGroup>
                          <InputGroup className="input-group-alternative">
                            <InputGroupAddon addonType="prepend">
                              <InputGroupText>
                                <svg className="bi bi-unlock-fill" width="1em" height="1em" viewBox="0 0 16 16" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                                  <path d="M.5 9a2 2 0 012-2h7a2 2 0 012 2v5a2 2 0 01-2 2h-7a2 2 0 01-2-2V9z"/>
                                  <path fillRule="evenodd" d="M8.5 4a3.5 3.5 0 117 0v3h-1V4a2.5 2.5 0 00-5 0v3h-1V4z" clipRule="evenodd"/>
                                </svg>
                              </InputGroupText>
                            </InputGroupAddon>
                            <Input required
                              placeholder="Password"
                              type="password"
                              autoComplete="off"
                              id="password"
                              name="password"
                              value={password}
                              onChange={this.handleChange}
                              invalid = { submitted && !password}
                            />
                            <FormFeedback>password is required!</FormFeedback>
                          </InputGroup>
                        </FormGroup>
                        <div className="custom-control custom-control-alternative custom-checkbox">
                          <input className="custom-control-input" id=" customCheckLogin" type="checkbox"/>
                          <label className="custom-control-label" htmlFor=" customCheckLogin">
                            <span>Remember me</span>
                          </label>
                        </div>
                        <div className="text-center">
                          <Button 
                            block 
                            className="my-4" 
                            color="default" 
                            size="lg"
                            href="/challenges" 
                            disabled={loading}
                            onClick={this.handleSubmit}
                          >
                            Signup
                          </Button>
                          {loading &&
                            <Spinner color="primary" />
                          }
                        </div>
                        <div className="text-center">
                          <a href="/login">
                            <h6>Login</h6>
                          </a>
                        </div>
                      </Form>
                    </CardBody>
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

export default Signup;
