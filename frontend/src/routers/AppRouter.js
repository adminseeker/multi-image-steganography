import Home from '../components/Home';
import React from 'react';
import { Router, Switch, Route } from 'react-router-dom';
import createHistory from 'history/createBrowserHistory';
const history = createHistory();
const AppRouter = () => {
  return (
    <Router history={history}>
      <Switch>
        <Route path='/' component={Home} exact={true} />
      </Switch>
    </Router>
  );
};
export default AppRouter;
