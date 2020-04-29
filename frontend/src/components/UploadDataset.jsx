import React from "react";
import { Link } from "react-router-dom";
import { Button, Card, CardImg, Modal, Form, FormGroup, CardHeader, CustomInput, CardBody, Input, Label, Spinner, UncontrolledAlert } from "reactstrap";

import { userService } from "../services/user_service";

class UploadDataset extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      ml_models: [],
      loading: false,
      error: false,
      errorMsg: '',
      'selectModel': ''
    }
    this.handleChange = this.handleChange.bind(this);
    this.handleDatasetUpload = this.handleDatasetUpload.bind(this);
  }

  componentDidMount() {
    var jwt = require('jsonwebtoken');
    var username = jwt.decode(JSON.parse(localStorage.getItem('user'))['token'])['username'];

    userService.getListOfModels(username)
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

  handleDatasetUpload(e) {
    e.preventDefault();
    const formData = new FormData();
    const fileField = document.querySelector('#datasetFile');
    formData.append('dataset', fileField.files[0]);
    const { selectModel } = this.state;
    if (!(selectModel && fileField.files[0])) {
      return;
    }
    this.setState({ loading: true });
    userService.uploadDataset(selectModel, formData)
      .then((resp) => {
          console.log('Success: '+resp);
          if(resp === 'uploaded'){
            this.setState({ loading: false });
            this.toggleModal("datasetModal");
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
    const { ml_models, error, errorMsg, loading } = this.state;
    return (
      <>
        <Card className="card-lift--hover bg-transparent shadow border-0 mt--300">
            <Link to="#" onClick={() => this.toggleModal("datasetModal")}>
            <CardImg
                alt="..."
                src={require("assets/img/upload_dataset_icon.png")}
            />
            </Link>

            {/*=============  Upload Dataset Modal  =============*/}
            <Modal
            className="modal-dialog-centered"
            size="sm"
            isOpen={this.state.datasetModal}
            toggle={() => this.toggleModal("datasetModal")}
            >
            <div className="modal-body p-0">
                <Card className="bg-secondary shadow border-0">
                <CardHeader className="bg-white pb-3">
                    <div className="text-muted text-center">
                    <h5>Upload Dataset</h5>
                    </div>
                </CardHeader>
                <CardBody className="px-lg-5 py-lg-5">
                    <div className="text-left">
                    <Form>
                        <FormGroup>
                        <Label for="selectModel">Select Model <span style={{color: '#FF0000'}}>*</span></Label>
                        <Input type="select" name="selectModel" id="selectModel" onChange={this.handleChange}>
                            <option value="">Select model</option>
                            {ml_models.map((model, index) => 
                            <option key={model._id} value={model._id}>
                                {index+1}. {model.name} ({model._id})
                            </option>
                            )}
                        </Input>
                        </FormGroup>
                        <FormGroup>
                        <CustomInput type="file" id="datasetFile" name="datasetFile" />
                        </FormGroup>
                    </Form>
                    </div>
                    { error ?
                    <div>
                        <UncontrolledAlert color="danger">
                        {errorMsg}
                        </UncontrolledAlert>
                    </div>
                    : <span></span>
                    }
                    <div className="text-center">
                    <Button block className="my-4" color="primary" size="lg"
                        disabled={loading}
                        onClick={this.handleDatasetUpload}
                    >
                        Upload
                    </Button>
                    {loading && <Spinner color="primary" /> }
                    </div>
                </CardBody>
                </Card>
            </div>
            </Modal>
      
        </Card>
      </>
    );
  }
}

export default UploadDataset;