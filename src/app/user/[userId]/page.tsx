'use client'
import React, { useEffect } from 'react'
import { useSession, signOut } from 'next-auth/react'
import { GoogleUserSession } from '@/app/types/GoogleUserSession'
import Image from 'next/image'
import { useRouter } from 'next/navigation'
import PinList from '@/app/components/Pins/PinList'

interface ProfileProps {
    params: any,
    searchParams: any,
}

const Profile:React.FC<ProfileProps> = ({params, searchParams}) => {
   const userEmail = params.userId.replace(/%40/g, '@')
   const { data: session } = useSession()
   const googleUserSession = session as GoogleUserSession
   const router = useRouter()
   const [listOfPins, setListOfPins] = React.useState<Array<any>>([])


   const onLogoutClick=()=>{
    signOut({ callbackUrl: '/' });
    localStorage.removeItem('userType');
  }

   const getArtistsPins = async () => {
    const userType = localStorage.getItem('userType')
    const response = await fetch(`/api/query/${userType}?email=${userEmail}`)
    const data = await response.json()
    return data
  }
  useEffect(() => {
    getArtistsPins().then((pins:Array<any>) => {
       setListOfPins(pins)
    })
  }
  , [userEmail])

  return (
    session?.user?.email === userEmail ? 
    <div>
      <div className='flex flex-col items-center'>
        <Image src={googleUserSession.user.image}
          alt='userImage'
          width={100}
          height={100}
          priority
          className='rounded-full'/>
          <h2 className='text-[30px]
          font-semibold'>
            {googleUserSession.user.name}
            </h2>
          <h2 className='text-gray-400'>User</h2>
          <h2 className='text-gray-400'>{googleUserSession.user.email}</h2>
          <div className='flex gap-4'>
          <button className='bg-gray-200
          p-2 px-3 font-semibold mt-5 rounded-full'>Share</button>

          {session?.user?.email== googleUserSession.user.email? <button className='bg-gray-200
          p-2 px-3 font-semibold mt-5 rounded-full'
          onClick={()=>onLogoutClick()}>Logout</button>:null}
      </div>
      </div>
      {listOfPins.length > 0 ? <PinList listOfPins={listOfPins} /> : <div>
        <h1 className='text-center text-3xl font-semibold mt-10'>
          No artpiece found
        </h1>
        </div>}
    </div> 
    : 
    <div>Not authorized</div>
  );
}

export default Profile