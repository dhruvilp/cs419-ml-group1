import React from "react";
import { Link } from "react-router-dom";
import { Button, Card, CardImg, Modal, Row, Col, Form, FormGroup, CardHeader, CustomInput, CardBody, InputGroup, Input, Label, Spinner, UncontrolledAlert, ModalHeader, ModalBody, ModalFooter } from "reactstrap";

import { userService } from "../services/user_service";

class UploadModel extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      ml_models: [],
      loading: false,
      error: false,
      errorMsg: '',
      uploadModelId: '',
      'challengeName': '',
      'challengeDescription': '',
      'selectModelType': '',
      'selectAttackType': '',
      'poolsSpec': '',
      'layersSpec': '',
      'selectInputColor': ''
    }
    this.handleChange = this.handleChange.bind(this);
    this.handleModelDescriptionUpload = this.handleModelDescriptionUpload.bind(this);
    this.handleModelUpload = this.handleModelUpload.bind(this);
  }

  componentDidMount() {
    userService.getListOfModels()
    .then((data) => {
      if(data){
        this.setState({
          ml_models : data
        });
      }
    })
    .catch((error) => {
      console.log(error);
    });
  }

  state = {};
  toggleModal = state => {
    this.setState({
      [state]: !this.state[state]
    });
  };

  handleChange(e) {
    const { name, value } = e.target;
    this.setState({ [name]: value });
  }

  handleModelDescriptionUpload() {
    const { challengeName, challengeDescription, selectModelType, selectAttackType, poolsSpec, layersSpec, selectInputColor } = this.state;
    if (!(challengeName && challengeDescription && selectModelType && selectAttackType)) {
      return;
    }
    this.setState({ loading: true });
    userService.publishNewModel(challengeName, challengeDescription, selectModelType, selectAttackType, poolsSpec, layersSpec, selectInputColor)
      .then((success) => {
          console.log(success);
          if(success){
            this.setState({ 
              loading: false,
              uploadModelId: success['_id']
            });
            this.toggleModal("toggleNested");
            return;
          }
        }
      )
      .catch((error) => {
        console.log('Error: '+error);
        this.setState({ 
          error: true, 
          loading: false,
          errorMsg: error
        });
        return;
      });
  }

  handleModelUpload(e) {
    e.preventDefault();
    const formData = new FormData();
    const fileField = document.querySelector('#modelFile');
    formData.append('model', fileField.files[0]);
    const { uploadModelId } = this.state;
    if (!(uploadModelId && fileField.files[0])) {
      return;
    }
    this.setState({ loading: true });
    userService.uploadTrainedModel(uploadModelId, formData)
      .then(
        (resp) => {
          console.log('Success: '+resp);
          if(resp === 'uploaded'){
            this.setState({ loading: false });
            this.toggleModal("toggleNested")
            this.toggleModal("modelModal");
          }
        }
      )
      .catch((error) => {
        console.log('Error: '+error);
        this.setState({ 
          error: true, 
          loading: false,
          errorMsg: error
        });
        return;
      });
  }

  render() {
    const { error, errorMsg, loading, challengeName, challengeDescription, poolsSpec, layersSpec } = this.state;
    return (
      <>
        <Card className="card-lift--hover bg-transparent shadow border-0 mt--300">
            <Link to="#" onClick={() => this.toggleModal("modelModal")}>
            <CardImg
                alt="..."
                src={require("assets/img/upload_model_icon.png")}
            />
            </Link>

            {/*=============  Upload Model Modal  =============*/}
            <Modal
                className="modal-dialog-centered"
                size="md"
                isOpen={this.state.modelModal}
                toggle={() => this.toggleModal("modelModal")}
                >
                <div className="modal-body p-0">
                    <Card className="bg-secondary shadow border-0">
                    <CardHeader className="bg-white pb-2">
                        <div className="text-muted text-center">
                        <h5>Upload Trained Model</h5>
                        </div>
                    </CardHeader>
                    <CardBody className="px-lg-5">
                        <Form role="form">
                        <FormGroup>
                            <Label for="challengeName">Name <span style={{color: '#FF0000'}}>*</span></Label>
                            <InputGroup className="input-group-alternative">
                            <Input
                                className="form-control-alternative"
                                placeholder="Challenge name..."
                                type="text"
                                id="challengeName"
                                name="challengeName"
                                value={challengeName}
                                onChange={this.handleChange}
                            />
                            </InputGroup>
                        </FormGroup>
                        <FormGroup>
                            <Label for="challengeDescription">Description <span style={{color: '#FF0000'}}>*</span></Label>
                            <InputGroup className="input-group-alternative">
                            <Input
                                className="form-control-alternative"
                                placeholder="Challenge description..."
                                rows="3"
                                type="textarea"
                                id="challengeDescription"
                                name="challengeDescription"
                                value={challengeDescription}
                                onChange={this.handleChange}
                            />
                            </InputGroup>
                        </FormGroup>
                        <Row>
                            <Col>
                            <FormGroup>
                                <Label for="selectModelType">Model Type <span style={{color: '#FF0000'}}>*</span></Label>
                                <Input type="select" name="selectModelType" id="selectModelType" onChange={this.handleChange}>
                                <option value="">Select model type</option>
                                <option value='cifar'>cifar</option>
                                <option value='mnist'>mnist</option>
                                </Input>
                            </FormGroup>
                            </Col>
                            <Col>
                            <FormGroup>
                                <Label for="selectAttackType">Attack Mode <span style={{color: '#FF0000'}}>*</span></Label>
                                <Input type="select" name="selectAttackType" id="selectAttackType" onChange={this.handleChange}>
                                <option value="">Select attack mode</option>
                                <option value='white'>white</option>
                                <option value='gray'>gray</option>
                                <option value='black'>black</option>
                                </Input>
                            </FormGroup>
                            </Col>
                        </Row>
                        <FormGroup>
                            <Label for="poolsSpec">Pools (optional)</Label>
                            <InputGroup className="input-group-alternative">
                            <Input
                                className="form-control-alternative"
                                placeholder="[List of JSON objects]..."
                                rows="3"
                                type="textarea"
                                id="poolsSpec"
                                name="poolsSpec"
                                value={poolsSpec}
                                onChange={this.handleChange}
                            />
                            </InputGroup>
                        </FormGroup>
                        <FormGroup>
                            <Label for="layersSpec">Layers (optional)</Label>
                            <InputGroup className="input-group-alternative">
                            <Input
                                className="form-control-alternative"
                                placeholder="[List of JSON objects]..."
                                rows="3"
                                type="textarea"
                                id="layersSpec"
                                name="layersSpec"
                                value={layersSpec}
                                onChange={this.handleChange}
                            />
                            </InputGroup>
                        </FormGroup>
                        <FormGroup>
                            <Label for="selectInputColor">Input Image Color (optional)</Label>
                            <Input type="select" name="selectInputColor" id="selectInputColor" onChange={this.handleChange}>
                            <option value="">Select (true: color, false: b/w)</option>
                            <option value='true'>true</option>
                            <option value='false'>false</option>
                            </Input>
                        </FormGroup>
                        <div className="text-center">
                            { error ?
                            <div>
                                <UncontrolledAlert color="danger">
                                {errorMsg}
                                </UncontrolledAlert>
                            </div>
                            : <span></span>
                            }
                            {loading && <Spinner color="primary" /> }
                            <Button block className="my-4" color="default" size="lg" onClick={() => this.handleModelDescriptionUpload()}>
                                Next
                            </Button>
                            <Modal isOpen={this.state.toggleNested} toggle={() => this.toggleModal("toggleNested")} onClosed={() => this.toggleModal("closeAll") ? () => this.toggleModal("modelModal") : undefined}>
                            <ModalHeader>Upload Trained Model</ModalHeader>
                            <ModalBody>
                                <div className="text-left">
                                <Form>
                                    <FormGroup>
                                    <CustomInput type="file" id="modelFile" name="modelFile" />
                                    </FormGroup>
                                </Form>
                                </div>
                            </ModalBody>
                            { error ?
                                <div>
                                <UncontrolledAlert color="danger">
                                    {errorMsg}
                                </UncontrolledAlert>
                                </div>
                                : <span></span>
                            }
                            <ModalFooter>
                                {loading && <Spinner color="primary" /> }
                                <Button color="secondary" onClick={() => this.toggleModal("toggleNested")}>Cancel</Button>{' '}
                                <Button color="primary" onClick={this.handleModelUpload}>Upload</Button>
                            </ModalFooter>
                            </Modal>
                        </div>
                        </Form>
                    </CardBody>
                    </Card>
                </div>
            </Modal>

        </Card>
      </>
    );
  }
}

export default UploadModel;