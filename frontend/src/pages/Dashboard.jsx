import React from "react";
import { Card, Container,Button ,ListGroup, ListGroupItem,Badge,FormText, Label,FormGroup,Input} from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar";
import CybneticsFooter from "components/CybneticsFooter";

class Dashboard extends React.Component {
  componentDidMount() {
    document.documentElement.scrollTop = 0;
    document.scrollingElement.scrollTop = 0;
    this.refs.main.scrollTop = 0;
  }
  render() {
    return (
      <>
        <CybneticsNavbar />
        <main ref="main">
        <section className="section-cybnetics-cover section-shaped my-0">
            <div className="shape shape-primary">
            
            </div>
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
                  <div className="text-center mt-5">
                  <FormGroup>
                  
        <Badge   pill variant color="primary"  >   Upload Dataset </Badge>

        <Input type="file" name="file" id="uploaddatasetfile" />
       
      </FormGroup>
      
      
      <FormGroup>
     
        <Badge  pill variant color="primary" >           Upload Trained Model</Badge>

        <Input type="file" name="file" id="uploadtrainedmodelfile" />
        
      </FormGroup>
                  </div>
                  
                  
                <div className="px-4">
                  <div className="text-left mt-5">
                    <FormText  color="black" size="lg">
                        Upload Datasets
                        </FormText>
                  </div>
                  
                </div>
                <Badge pill variant color="primary" border-radius="18" size="lg" >Upload Datasets</Badge>
                  
                  
                <ListGroup>
      <ListGroupItem>First<Button color="primary" className="float-right" >Delete </Button>
      <Button color="primary" className="float-right" >Edit </Button>
                          </ListGroupItem>
      <ListGroupItem>Second<Button color="primary" className="float-right" >Delete </Button>
      <Button color="primary" className="float-right" >Edit </Button></ListGroupItem>
      <ListGroupItem>Third<Button color="primary" className="float-right" >Delete </Button>
      <Button color="primary" className="float-right" >Edit </Button></ListGroupItem>
      <ListGroupItem>Fourth<Button color="primary" className="float-right" >Delete </Button>
      <Button color="primary" className="float-right" >Edit </Button></ListGroupItem>
      <ListGroupItem>5555555<Button color="primary" className="float-right" >Delete </Button>
      <Button color="primary" className="float-right" >Edit </Button> </ListGroupItem>
    </ListGroup> 
                  
  
    <Badge pill variant color="primary" border-radius="18" size="lg" >Uploaded Models</Badge>
          
    <ListGroup>
      <ListGroupItem>First<Button color="primary" className="float-right" >Delete </Button>
      <Button color="primary" className="float-right" >Edit </Button>
                          </ListGroupItem>
      
    </ListGroup>        
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  <div className="text-center mt-5">
                    <Container>
                      <h6>
                        Only Admin Access This Page
                      </h6>
                    </Container>
                  </div>
                </div>
              </Card>
            </Container>
          </section>
          
    );
          
        </main>
        <CybneticsFooter />
      </>
    );
  }
}

export default Dashboard;
