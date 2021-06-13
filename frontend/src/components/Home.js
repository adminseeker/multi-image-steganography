import React from 'react';
import Header from './Header';
import Upload from './Upload';
const Home = () => {
  return (
    <div>
      <div>
        <Header />
      </div>
      <br />
      <br />
      <br />
      <br />
      <br />
      <h3 style={{textAlign:"center"}}>Upload One Container Image and Maximum Three Secret Images</h3>
      <br />
      <br />
      <div>
        <Upload />
      </div>
    </div>
  );
};

export default Home;
