import { DropzoneArea } from "material-ui-dropzone";
import React, { useEffect, useState } from "react";
import Button from "@material-ui/core/Button";
import CloudUploadIcon from "@material-ui/icons/CloudUpload";
import axios from "axios";
import Cookies from "universal-cookie";
import LockIcon from "@material-ui/icons/Lock";
import LockOpenIcon from "@material-ui/icons/LockOpen";
import Grid from "@material-ui/core/Grid";
import { MuiThemeProvider, createMuiTheme } from "@material-ui/core/styles";

import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
  },
  paper: {
    padding: theme.spacing(2),
    textAlign: "center",
    color: theme.palette.text.secondary,
  },
}));


const theme = createMuiTheme({
  overrides: {
    MuiDropzoneSnackbar: {
      errorAlert: {
        backgroundColor: "#FAA",
        color: "#000"
      },
      successAlert: {
        backgroundColor: "#AFA",
        color: "#000"
      },
     
    },
     MuiDropzonePreviewList:{
        removeButton:{
          right:"40px"
        }
      },
  }
});


const Upload = () => {
  const classes = useStyles();
  const [resultsButtons, setResultsButtons] = useState(false);
  const [uploadButton, setUploadButton] = useState(false);
  const [clickedEncode, setClickedEncode] = useState(false);
  const [clickedDecode, setClickedDecode] = useState(false);
  const [img1, setImg1] = useState(false);
  const [img2, setImg2] = useState(false);
  const [img3, setImg3] = useState(false);
  const [gridSpace, setGridSpace] = useState(6);
  const [encodeRes, setEncodeRes] = useState({ original: "", encoded: "" });
  const [decodeRes, setDecodeRes] = useState({
    decoded1: "",
    decoded2: "",
    decoded3: "",
    secret1: "",
    secret2: "",
    secret3: "",
  });
  const [file, setFile] = useState({
    container: undefined,
    input1: undefined,
    input2: undefined,
    input3: undefined,
  });

  useEffect(() => {
    console.log("file", file);
    console.log(
      "bool",
      file.container && file.input1 && file.input2 && file.input3
    );
    if (file.container && (file.input1 || file.input2 || file.input3)) {
      setUploadButton(true);
    } else {
      setUploadButton(false);
    }
  }, [file, setUploadButton, uploadButton]);

  const sendToServer = async (data) => {
    try {
      const config = {
        headers: {
          "Content-Type": "application/json",
        },
      };
      const res = await axios.post("/api/upload", data, config);
      const cookies = new Cookies();
      cookies.set("fid", res.data.fid);
      setResultsButtons(true);
      console.log(res.data.fid);
    } catch (error) {
      console.log(error);
    }
  };

  const toBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });
  const onClickUpload = async () => {
    let t = await toBase64(file.container);
    let container = t.toString().split(",")[1];
    let input1;
    let input2;
    let input3;
    let imgCount = 0;
    setResultsButtons(false);
    if (file.input1) {
      t = await toBase64(file.input1);
      input1 = t.toString().split(",")[1];
      imgCount += 1;
    }
    if (file.input2) {
      t = await toBase64(file.input2);
      input2 = t.toString().split(",")[1];
      imgCount += 1;
    }
    if (file.input3) {
      t = await toBase64(file.input3);
      input3 = t.toString().split(",")[1];
      imgCount += 1;
    }
    const cookies = new Cookies();
    cookies.set("imgCount", imgCount);

    setUploadButton(false);
    if (input1 && !input2 && !input3) {
      input2 = input1;
      input3 = input1;
    } else if (input1 && input2 && !input3) {
      input3 = input1;
    }
    sendToServer({ container, input1, input2, input3 });
  };

  const onClickEncode = async () => {
    setClickedDecode(false);
    const cookies = new Cookies();
    let fid = cookies.get("fid");
    try {
      const config = {
        headers: {
          "Content-Type": "application/json",
        },
      };
      let data = { fid };
      let res = await axios.post("/api/encode", data, config);
      setEncodeRes({
        ...encodeRes,
        original: res.data.original,
        encoded: res.data.encoded,
      });
      setClickedEncode(true);
      console.log(res);
    } catch (error) {
      console.log(error);
    }
  };
  const onClickDecode = async () => {
    setClickedEncode(false);
    const cookies = new Cookies();
    let fid = cookies.get("fid");
    try {
      const config = {
        headers: {
          "Content-Type": "application/json",
        },
      };
      let data = { fid };
      let imgCount = cookies.get("imgCount");
      if (imgCount >= 1) {
        setImg1(true);
      }
      if (imgCount >= 2) {
        setImg2(true);
      }
      if (imgCount == 3) {
        setImg3(true);
      }
      if (imgCount==1) setGridSpace(6);
      if (imgCount==2) setGridSpace(3);
      if (imgCount==3) setGridSpace(2);
      let res = await axios.post("/api/decode", data, config);
      setDecodeRes({
        ...decodeRes,
        decoded1: res.data.decoded1,
        decoded2: res.data.decoded2,
        decoded3: res.data.decoded3,
        secret1: res.data.secret1,
        secret2: res.data.secret2,
        secret3: res.data.secret3,
      });
      setClickedDecode(true);
      console.log(res);
    } catch (error) {
      console.log(error);
    }
  };
  return (
    <div>
      {
        <Grid container spacing={1}>
          <Grid item xs={2}>
          </Grid>
        <Grid item xs={8}>
          <MuiThemeProvider theme={theme}>
        <DropzoneArea
          acceptedFiles={["image/jpeg", "image/jpg"]}
          dropzoneText={"Drag and drop an container image here or click"}
          onChange={async (files) => {
            setFile({ ...file, container: files[0] });
          }}
          onAdd={()=>{
            setResultsButtons(false)
            setImg1(false)
            setImg2(false)
            setImg3(false)
            
          }}
          onDrop={()=>{
            setResultsButtons(false)
            setImg1(false)
            setImg2(false)
            setImg3(false)
            
          }}
          onDelete={()=>{
            setResultsButtons(false)
            setImg1(false)
            setImg2(false)
            setImg3(false)
            
          }}
          
          filesLimit={1}
        />
        </MuiThemeProvider >
        </Grid >
        <Grid item xs={2}>
          </Grid>
        </Grid >
      }
      {
        <Grid container spacing={1}>
        <Grid item xs={2}>
        </Grid>
      <Grid item xs={8}>
      <MuiThemeProvider theme={theme}>
        <DropzoneArea
          acceptedFiles={["image/jpeg", "image/jpg"]}
          dropzoneText={"Drag and drop secret images here or click"}
          onChange={async (files) => {
            setFile({
              ...file,
              input1: files[0],
              input2: files[1],
              input3: files[2],
            });
          }}
          onDelete={()=>{
            setResultsButtons(false)
            setImg1(false)
            setImg2(false)
            setImg3(false)
            
          }}
          onAdd={()=>{
            setResultsButtons(false)
            setImg1(false)
            setImg2(false)
            setImg3(false)
            
          }}
          onDrop={()=>{
            setResultsButtons(false)
            setImg1(false)
            setImg2(false)
            setImg3(false)
            
          }}
          filesLimit={3}
        />
        </MuiThemeProvider >
        </Grid >
        <Grid item xs={2}>
          </Grid>
        </Grid >
      }
      <br />
      <div className={classes.root}>
      <Grid container spacing={1}>
          <Grid item xs={2}>
          </Grid>
        <Grid item xs={8}>
        <Grid container spacing={3}>
          <Grid item xs={4}>
            {
              <Button
                variant="contained"
                color="default"
                startIcon={<CloudUploadIcon />}
                onClick={onClickUpload}
                disabled={!uploadButton}
                className={classes.paper}
              >
                Upload
              </Button>
            }
          </Grid>
          <Grid item xs={4} style={{ textAlign: "center" }}>
            <Button
              variant="contained"
              color="secondary"
              startIcon={<LockIcon />}
              onClick={onClickEncode}
              disabled={!resultsButtons}
              className={classes.paper}
            >
              View Encode Results
            </Button>
          </Grid>
          <Grid item xs={4} style={{ textAlign: "right" }}>
            <Button
              variant="contained"
              color="primary"
              startIcon={<LockOpenIcon />}
              onClick={onClickDecode}
              disabled={!resultsButtons}
              className={classes.paper}
            >
              View Decode Results
            </Button>
          </Grid>
        </Grid>
        <Grid item xs={2}>
          </Grid>
          </Grid>
        </Grid >
      </div>
            <br />
      {clickedEncode && (
        <div className={classes.root}>
          <Grid container spacing={1}>
            <Grid item xs={3} className={classes.paper}></Grid>
            <Grid item xs={3} className={classes.paper} style={{ textAlign: "right" }}>
              <img src={`data:image/jpeg;base64,${encodeRes.original}`} />
              <p style={{ textAlign: "right" }}>Original</p>
            </Grid>
            <Grid item xs={3} className={classes.paper} style={{ textAlign: "left" }}>
              <img src={`data:image/jpeg;base64,${encodeRes.encoded}`} />
              <p style={{ textAlign: "left" }}>Encoded</p>
            </Grid>
            <Grid item xs={3} className={classes.paper}></Grid>
          </Grid>
        </div>
      )}
      {clickedDecode && (
        <div className={classes.root}>
          <Grid container spacing={1}>
            {img1 && <Grid item xs={gridSpace} className={classes.paper} style={{ textAlign: "right" }}>
              
                <img src={`data:image/jpeg;base64,${decodeRes.secret1}`} />
                <p style={{ textAlign: "right" }}>secret</p>
              
            </Grid>}
            {img1 &&<Grid item xs={gridSpace} className={classes.paper} style={{ textAlign: "left" }}>
                <img src={`data:image/jpeg;base64,${decodeRes.decoded1}`} />
                <p style={{ textAlign: "left" }}>decoded</p>
                
            </Grid>}
            {img2 &&<Grid item xs={gridSpace} className={classes.paper} style={{ textAlign: "right" }}>
                <img src={`data:image/jpeg;base64,${decodeRes.secret2}`} />
                <p style={{ textAlign: "right" }}>secret</p>
                
            </Grid>}
            {img2 && <Grid item xs={gridSpace} className={classes.paper} style={{ textAlign: "left" }}>
                <img src={`data:image/jpeg;base64,${decodeRes.decoded2}`} />
                <p style={{ textAlign: "left" }}>decoded</p>
            </Grid>}
            {img3 && <Grid item xs={gridSpace} className={classes.paper} style={{ textAlign: "right" }}>
                <img src={`data:image/jpeg;base64,${decodeRes.secret3}`} />
                <p style={{ textAlign: "right" }}>secret</p>
            </Grid>}
            {img3 && <Grid item xs={gridSpace} className={classes.paper} style={{ textAlign: "left" }}>
                <img src={`data:image/jpeg;base64,${decodeRes.decoded3}`} />
                <p style={{ textAlign: "left" }}>decoded</p>
            </Grid>}
          </Grid>
        </div>
      )}
    </div>
  );
};

export default Upload;
