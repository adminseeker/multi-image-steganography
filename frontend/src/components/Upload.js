import { DropzoneArea } from 'material-ui-dropzone';
import React, { useState } from 'react';
import Button from '@material-ui/core/Button';
import CloudUploadIcon from '@material-ui/icons/CloudUpload';
const Upload = () => {
  const [file, setFile] = useState({
    container: '',
    input1: '',
    input2: '',
    input3: '',
  });
  const toBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });
  const onClick = async () => {
    // console.log(file);
    console.log(await toBase64(file.container));
  };
  return (
    <div>
      <DropzoneArea
        acceptedFiles={['image/jpeg', 'image/jpg']}
        dropzoneText={'Drag and drop an container image here or click'}
        onChange={(files) => {
          setFile({ ...file, container: files[0] });
        }}
        filesLimit={1}
      />
      <DropzoneArea
        acceptedFiles={['image/jpeg', 'image/jpg']}
        dropzoneText={'Drag and drop secret images here or click'}
        onChange={(files) => {
          setFile({
            ...file,
            input1: files[0],
            input2: files[1],
            input3: files[2],
          });
        }}
        filesLimit={3}
      />
      <Button
        variant='contained'
        color='default'
        startIcon={<CloudUploadIcon />}
        onClick={onClick}
      >
        Upload
      </Button>
    </div>
  );
};

export default Upload;
