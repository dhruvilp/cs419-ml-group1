import React from "react";
import { Link } from "react-router-dom";
import { Button, Card, CardImg, Container, Modal, Row, Col, Form, FormGroup, CardHeader, CustomInput, CardBody, InputGroup, Input } from "reactstrap";

class UploadButtons extends React.Component {

  state = {};
  toggleModal = state => {
    this.setState({
      [state]: !this.state[state]
    });
  };

  render() {
    return (
      <>
      <Container className="container-md">
        <Row>
          <Col className="mb-5 mb-md-0" md="6">
            <Card className="card-lift--hover bg-transparent shadow border-0 mt--300">
              <Link onClick={() => this.toggleModal("datasetModal")}>
                <CardImg
                  alt="..."
                  src={require("assets/img/upload_dataset_icon.png")}
                />
              </Link>
            </Card>
          </Col>
          <Col className="mb-5 mb-lg-0" md="6">
            <Card className="card-lift--hover bg-transparent shadow border-0 mt--300">
              <Link onClick={() => this.toggleModal("modelModal")}>
                <CardImg
                  alt="..."
                  src={require("assets/img/upload_model_icon.png")}
                />
              </Link>
            </Card>
          </Col>
        </Row>

        {/*=============  Upload Buttons Modals  =============*/}
        <Modal
          className="modal-dialog-centered"
          size="sm"
          isOpen={this.state.datasetModal}
          toggle={() => this.toggleModal("datasetModal")}
        >
          <div className="modal-body p-0">
            <Card className="bg-secondary shadow border-0">
              <CardHeader className="bg-white pb-3">
                <div className="text-muted text-center mb-3">
                  <h5>Upload Dataset</h5>
                </div>
                <div className="text-left">
                  <Form>
                    <FormGroup>
                      <CustomInput type="file" id="datasetFileBrowser" name="customFile" />
                    </FormGroup>
                  </Form>
                </div>
              </CardHeader>
              <CardBody className="px-lg-5 py-lg-5">
                <div className="text-center">
                  <Button block className="my-4" color="primary" size="lg"
                    onClick={() => this.toggleModal("datasetModal")}
                  >
                    Upload
                  </Button>
                </div>
              </CardBody>
            </Card>
          </div>
        </Modal>

        <Modal
          className="modal-dialog-centered"
          size="sm"
          isOpen={this.state.modelModal}
          toggle={() => this.toggleModal("modelModal")}
        >
          <div className="modal-body p-0">
            <Card className="bg-secondary shadow border-0">
              <CardHeader className="bg-white pb-3">
                <div className="text-muted text-center mb-3">
                  <h5>Upload Trained Model</h5>
                </div>
                <div className="text-left">
                  <Form>
                    <FormGroup>
                      <CustomInput type="file" id="mlModelBrowser" name="customFile" />
                    </FormGroup>
                  </Form>
                </div>
              </CardHeader>
              <CardBody className="px-lg-5 py-lg-5">
                <Form role="form">
                  <FormGroup>
                    <InputGroup className="input-group-alternative">
                      <Input
                        className="form-control-alternative"
                        placeholder="Challenge name..."
                        type="text"
                      />
                    </InputGroup>
                  </FormGroup>
                  <FormGroup>
                    <InputGroup className="input-group-alternative">
                      <Input
                        className="form-control-alternative"
                        placeholder="Challenge description..."
                        rows="3"
                        type="textarea"
                      />
                    </InputGroup>
                  </FormGroup>
                  <Col>
                    <div className="custom-control custom-radio mb-3">
                      <input
                        className="custom-control-input"
                        defaultChecked
                        id="customRadio1"
                        name="custom-radio-1"
                        type="radio"
                      />
                      <label className="custom-control-label" htmlFor="customRadio1">
                        <span>WHITE BOX MODE</span>
                      </label>
                    </div>
                    <div className="custom-control custom-radio mb-3">
                      <input
                        className="custom-control-input"
                        id="customRadio2"
                        name="custom-radio-1"
                        type="radio"
                      />
                      <label className="custom-control-label" htmlFor="customRadio2">
                        <span>BLACK BOX MODE</span>
                      </label>
                    </div>
                    <div className="custom-control custom-radio mb-3">
                      <input
                        className="custom-control-input"
                        id="customRadio3"
                        name="custom-radio-1"
                        type="radio"
                      />
                      <label className="custom-control-label" htmlFor="customRadio3">
                        <span>GERY BOX MODE</span>
                      </label>
                    </div>
                  </Col>
                  <div className="text-center">
                    <Button block className="my-4" color="primary" size="lg"
                      onClick={() => this.toggleModal("modelModal")}
                    >
                      Upload
                    </Button>
                  </div>
                </Form>
              </CardBody>
            </Card>
          </div>
        </Modal>

      </Container>
      </>
    );
  }
}

export default UploadButtons;