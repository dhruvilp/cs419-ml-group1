import React from "react";
import { Container, Row, Col } from "reactstrap";
import UploadDataset from "./UploadDataset";
import UploadModel from "./UploadModel";

class UploadButtons extends React.Component {
  render() {
    return (
      <>
      <Container className="container-md">
        <Row>
          <Col className="mb-5 mb-md-0" md="6">
            <UploadModel />
          </Col>
          <Col className="mb-5 mb-lg-0" md="6">
            <UploadDataset />
          </Col>
        </Row>
      </Container>
      </>
    );
  }
}

export default UploadButtons;