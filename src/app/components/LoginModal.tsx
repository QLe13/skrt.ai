import React, { useState } from 'react';
import { signIn } from 'next-auth/react';
import { FaGoogle } from "react-icons/fa";
import { IoMdCloseCircle } from "react-icons/io";

interface LoginModalProps {
    toggleLoginModal: () => void; // Updated type for clarity and correctness
}

const LoginModal: React.FC<LoginModalProps> = ({ toggleLoginModal }) => {
    const [userType, setUserType] = useState('');

    const handleUserType = (type: string) => {
        setUserType(type);
    }
    const logIn = async (provider: string) => {
        if (userType === '') {
            alert('Please select user type')
            return
        }
        localStorage.setItem('userType', userType);
        await signIn(provider);
    }






    return (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full" id="my-modal"> {/* Overlay */}

            <div className="absolute top-5 right-10">
                <IoMdCloseCircle
                    size={30}
                    className="text-black hover:text-gray-700 cursor-pointer"
                    onClick={() => toggleLoginModal()}
                />
            </div>
            <div className="relative top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white p-5 rounded-lg shadow-lg w-96"> {/* Modal box */}
                <div className="text-center">
                    <button
                        className="flex justify-center items-center gap-2 mx-auto hover:bg-gray-300 p-2 px-4 rounded-full"
                        onClick={() => logIn('google')}>
                        <FaGoogle size={20} className='text-gray-500' />
                        Log in with Google
                    </button>
                </div>
                <button className={`flex justify-center items-center gap-2 mx-auto p-2 px-4 rounded-full ${userType === 'user' ? 'bg-black text-white' : 'hover:bg-gray-300'}`}
                onClick={() => handleUserType('user')}
                >
                        User
                </button>
                <button className={`flex justify-center items-center gap-2 mx-auto p-2 px-4 rounded-full ${userType === 'artist' ? 'bg-black text-white' : 'hover:bg-gray-300'}`}
                onClick={() => handleUserType('artist')}
                >
                        Artist
                </button>
            </div>
        </div>
    );
};

export default LoginModal;
