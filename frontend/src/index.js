import React from "react";
import ReactDOM from "react-dom";
import { BrowserRouter, Route, Switch, Redirect } from "react-router-dom";
import * as serviceWorker from './serviceWorker';

import "assets/css/custom.css";
import "assets//font-awesome/css/font-awesome.min.css";
import "assets/scss/argon-design-system-react.scss?v1.1.0";

import Login from "pages/Login";
import Signup from "pages/Signup";
import Leaderboard from "pages/Leaderboard";
import Openground from "pages/Openground";
import Challenges from "pages/Challenges";
import Dashboard from "pages/Dashboard";
import { RestrictedRoute } from "components/RestrictedRoute";

ReactDOM.render(
  <React.StrictMode>
    <BrowserRouter>
      <Switch>
        <Route 
          path="/" 
          exact 
          render={props => <Login {...props} />} 
        />
        <RestrictedRoute 
          exact 
          path="/challenges" 
          component={Challenges} 
        />
        <RestrictedRoute 
          exact 
          path="/leaderboard" 
          component={Leaderboard} 
        />
        <RestrictedRoute 
          exact 
          path="/openground" 
          component={Openground} 
        />
        <RestrictedRoute 
          exact 
          path="/dashboard" 
          component={Dashboard} 
        />
        <RestrictedRoute 
          exact 
          path="/dashboard" 
          component={Dashboard} 
        />
        {/* <Route 
          path="/challenges" 
          exact 
          render={props => <Challenges {...props} />} 
        /> */}
        {/* <Route
          path="/leaderboard"
          exact
          render={props => <Leaderboard {...props} />}
        /> */}
        {/* <Route
          path="/openground"
          exact
          render={props => <Openground {...props} />}
        /> */}
        {/* <Route
          path="/dashboard"
          exact
          render={props => <Dashboard {...props} />}
        /> */}
        <Route 
          path="/login" 
          exact 
          render={props => <Login {...props} />} 
        />
        <Route 
          path="/signup" 
          exact 
          render={props => <Signup {...props} />} 
        />
        <Redirect to="/" />
      </Switch>
    </BrowserRouter>,
  </React.StrictMode>,
  document.getElementById("root")
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();