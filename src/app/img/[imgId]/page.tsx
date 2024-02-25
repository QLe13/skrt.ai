'use client'
import React from 'react'
import { HiArrowSmallLeft } from "react-icons/hi2";
import Image from 'next/image';
import { useRouter } from 'next/navigation';

interface ImageDetailProps {
    params: {
        imgId: string
    },
    searchParams: {
        Details: string
    },
}

const ImageDetail: React.FC<ImageDetailProps> = ({params, searchParams}) => {
    const router = useRouter()
    const imgId = params.imgId
    const details = JSON.parse(searchParams.Details)
    const image_url = details.image_url
    const title = details.title
  return (
    <div className=' bg-white p-3 md:p-12 rounded-2xl md:px-24 lg:px-36 flex gap-2'>
        <HiArrowSmallLeft className='text-[80px] font-bold ml-[-50px] 
        cursor-pointer hover:bg-gray-200 rounded-full p-2'
        onClick={()=>router.back()}/>
    <div className='grid grid-cols-1 lg:grid-cols-2 md:gap-10 shadow-xl rounded-2xl p-3 md:p-7 lg:p-12 xl:pd-16' >
        <div>
            <Image src={image_url}
            alt={title}
            width={1000}
            height={1000}
            className='rounded-2xl'/>
        </div>
        <div className="">
            <div>
                <h2 className='text-[30px] font-bold mb-10'>Artwork Name: {title}</h2>
                <h2 className='mt-10'>ID: {imgId}</h2>
                <button className='p-2 bg-[#e9e9e9] px-5 text-[23px]
                mt-10 rounded-full hover:scale-105 transition-all'
                onClick={()=>window.open(image_url)}>Open Url</button>
            </div>
        </div>
    </div>
    </div>
  )
}

export default ImageDetail;