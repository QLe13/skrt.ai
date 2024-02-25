import React from 'react'
import PinItem from './PinItem'

interface PinListProps {
    listOfPins:Array<any>
}


const PinList:React.FC<PinListProps> = ({listOfPins}) => {
  return (
    <div className='mt-7 px-2 md:px-5
     columns-2 md:columns-3
     lg:columns-4 mb-4
     xl:columns-5 space-y-6 mx-auto
     '>
        {listOfPins.map((item,index)=>(
               <PinItem key={index}  Pin={item} />
        ))}
    </div>
  )
}

export default PinList