"use client"
import React from 'react'
import Image from 'next/image'
import logoImage from '../../../public/skrt.ai.png'
import { HiChat } from 'react-icons/hi'
import { useSession } from 'next-auth/react';
import { useState, useCallback, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { GrPaint } from "react-icons/gr";

import LoginModal from './LoginModal'



const Header = () => {
    const {data:session} = useSession()
    const [showModal, setShowModal] = useState(false)
    const router = useRouter()
    const userType = typeof window !== 'undefined' ? localStorage.getItem('userType') : null;
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
    setIsClient(true);
    }, []);



    const handleProfileRedirect = () => {
        if (!session?.user) {
            alert('Please login to continue')
            return
        }
        router.push('/'+userType+'/'+session?.user?.email)
    }

    const toggleLoginModal = useCallback(() => {
        setShowModal(!showModal)
    }, [showModal])

    
    return (
        <div className='flex justify-between gap-3 md:gap-2 items-center p-6 w-full z-50 bg-white sticky top-0'>
            <div className="flex items-center gap-3 cursor-pointer hover:bg-gray-300 rounded-md p-3 " onClick={() => router.push('/')}>
                <Image
                    src={logoImage}
                    alt='logo'
                    priority
                    className='w-12 h-19'
                />
                <span className='text-xl font-semibold'>sKrt.ai</span>
            </div>
            <button
            className=' hover:bg-gray-300 p-2 px-4 rounded-full'
            onClick={() => router.push('/')}
            >Home</button>
            <div className='flex gap-3 bg-[#e9e9e9] p-2 items-center rounded-full w-5/12 hidden md:flex'>
                <GrPaint />
                <input
                className='flex bg-transparent outline-none border-none w-full'
                type='text' placeholder='Dream about...'/>
                
            </div>
            <GrPaint  className='md:hidden'/>
            {!session && <button className='bg-black text-white hover:bg-gray-700 p-2 px-4 rounded-full'
            onClick={toggleLoginModal}
            >Get Creative</button>}
            {isClient && userType==='user' && <button className='bg-black text-white hover:bg-gray-700 p-2 px-4 rounded-full'>Dream</button>}
            <button className='flex hover:bg-gray-300 p-2 px-4 rounded-full'
            >About</button>
            <button className='flex hover:bg-gray-300 p-2 px-4 rounded-full'
            >Artist</button>
            <HiChat size={50} className ='hover:bg-gray-300 rounded-full cursor-pointer text-gray-500' />
            {session?.user? 
            <Image 
            src={session?.user?.image as string} 
            priority
            alt='user' width={40} height={40} 
            className='rounded-full cursor-pointer' 
            onClick={() => handleProfileRedirect()}
            />
            :
            <></>}
            {showModal && <LoginModal toggleLoginModal={toggleLoginModal} />}
        </div>
    )
}

export default Header
