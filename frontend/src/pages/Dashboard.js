import React from "react";
import { Card, Container, } from "reactstrap";

import CybneticsNavbar from "components/CybneticsNavbar.js";
import CybneticsFooter from "components/CybneticsFooter.js";

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
            <div className="shape shape-primary"></div>
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
                    <h3>
                        Dashboard
                    </h3>
                  </div>
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
        </main>
        <CybneticsFooter />
      </>
    );
  }
}

export default Dashboard;
