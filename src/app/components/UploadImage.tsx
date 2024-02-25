import React, { useState } from 'react'
import { HiArrowUpCircle } from "react-icons/hi2";

interface UploadImageProps {
    setFile:(file: File | null) => void;
}
const UploadImage:React.FC<UploadImageProps> =({setFile}) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);


  return (
    <div className='h-[450px] bg-[#e9e9e9]
    rounded-lg'>
        
        <label className='m-5 flex flex-col justify-center items-center
        cursor-pointer h-[90%] 
        border-[2px] border-gray-300 border-dashed rounded-lg text-gray-600 '>
           
          {!selectedFile?
          <div className='flex items-center flex-col'>
           <HiArrowUpCircle className='text-[22px]'/>   
            <h2 className=' font-semibold'>Click to Upload</h2>
            </div>
            :null}
            {selectedFile?
            <img src={window.URL.createObjectURL(selectedFile)}
            alt='selected-image'
            width={500}
            height={800}
            className='object-contain h-[90%]'
            />:null}
            <input id="dropzone-file" type="file"
             className="hidden"  
             onChange={(e)=>{
                if (e.target.files && e.target.files[0]) {
                    const file = e.target.files[0];
                    setFile(file);
                    setSelectedFile(file);
                } else {
                    setFile(null);
                    setSelectedFile(null);
                }
                }}/>
        </label>
    </div>
  )
}

export default UploadImage