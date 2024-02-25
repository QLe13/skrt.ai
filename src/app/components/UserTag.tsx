import React from 'react'
import Image from 'next/image';
import { Session } from 'next-auth';

interface UserTagProps {
    session: Session | null
}

const UserTag:React.FC<UserTagProps> = ({session}) => {
    const user = session?.user
  return (
    <div className=''>
       {user?
       <div className='flex gap-3 
       items-center'>
       <Image src={user.image as string} 
       alt='userImage'
       width={45}
       height={45}
       priority
       className='rounded-full'/>
       <div>
        <h2 className='text-[14px] font-medium'>{user.name}</h2>
        <h2 className='text-[12px]'>{user.email}</h2>

        </div>
       </div>
       :null}
    </div>
  )
}

export default UserTag