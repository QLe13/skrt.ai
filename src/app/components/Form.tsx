"use client"
import React,{useState} from 'react'
import UploadImage from './UploadImage'
import { useSession} from "next-auth/react"
import UserTag  from './UserTag'
import { useRouter } from 'next/navigation'
import Image from 'next/image'


const Form = () => {
    const {data:session}=useSession();
    const [title, setTitle] = useState<string>(''); 
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const router = useRouter();

    const onSave = async () => {
        setLoading(true);
      
        if (!file) {
          alert("No file selected");
          setLoading(false);
          return;
        }
      
        const formData = new FormData();
        const user = session?.user;
        formData.append("file", file);
        formData.append("title", title);
        formData.append("user", user?.email as string);
        
        // Append any other fields as needed
      
        try {
          const response = await fetch('/api/query', {
            method: 'POST',
            body: formData,
          });
      
          if (!response.ok) throw new Error("Upload failed");
      
          // Handle success response, possibly redirect or clear form
          const userType = localStorage.getItem('userType')
          router.push('/'+userType+'/'+session?.user?.email)
        } catch (error) {
          console.error("Upload error:", error);
        } finally {
          setLoading(false);
        }
    };
      


   
   
  return (
    <div className=' bg-white p-16 rounded-2xl '>
        <div className='flex justify-end mb-6'>
            <button onClick={onSave} className='bg-red-500 p-2 text-white font-semibold px-3 rounded-lg'>
            {loading ? (
                <Image
                src="/loading-indicator.png"
                width={30}
                height={30}
                alt='loading'
                className='animate-spin'
                />
            ) : (
                <span>Save</span>
            )}
            </button>
        </div>
        <div className='grid grid-cols-1 lg:grid-cols-3 gap-10 items-center'>
            <UploadImage setFile={(file: any) => setFile(file)} />
            <div className="col-span-2 flex flex-col items-center"> {/* Adjust this line */}
                <div className='w-[100%] flex flex-col items-center'> {/* Wrapper with Flexbox styles */}
                    <input
                        type="text"
                        placeholder='Add your title'
                        onChange={(e) => setTitle(e.target.value)}
                        className='text-[35px] outline-none font-bold w-full border-b-[2px] border-gray-400 placeholder-gray-400'
                    />
                    <h2 className='text-[12px] mb-8 w-full text-gray-400'>
                        The first 40 Characters are what usually show up in feeds
                    </h2>
                    <UserTag session={session} />
                </div>
            </div>
        </div>
    </div>

  )
}

export default Form