import Image from 'next/image'
import React from 'react'
import { useRouter } from 'next/navigation'



interface PinItemProps {
    Pin: {
        id: number,
        artist_id: number,
        title: string,
        created_at: string,
        image_url: string
    }
}

const PinItem:React.FC<PinItemProps> = ({Pin}) => {
  const router=useRouter();

  return (
    <div className='relative'>
       <div className="relative 
       before:absolute
       before:h-full before:w-full
       before:rounded-3xl
       before:z-10
       hover:before:bg-gray-600 
       before:opacity-50
       cursor-pointer
       " 
       onClick={()=>{
        const infor = {
          title: Pin.title,
          image_url: Pin.image_url
        }
        router.push("/img/"+Pin.id+"?Details="+JSON.stringify(infor))
        }}>
       
        <Image src={Pin.image_url}
        alt={Pin.title}
        width={500}
        priority
        height={500}
        className='rounded-3xl 
        cursor-pointer relative z-0'
        />
       </div>
        <h2 className='font-bold 
        text-[18px] mb-1 mt-2 line-clamp-2'>{Pin.title}</h2>
        <div className=''>

       <div className='flex gap-3 
       items-center'>
       <div>
        <h2 className='text-[14px] font-medium'>Artwork ID: {Pin.id}</h2>
        <h2 className='text-[12px]'>Create at: {Pin.created_at}</h2>

        </div>
       </div>
      
    </div>
    </div>
  )
}

export default PinItem