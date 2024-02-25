'use client'
import React from "react";
import { useSession } from "next-auth/react";
import Form from "../components/Form";

const ArtBuilder = () => {
  const { data: session, status } = useSession();
  const loading = status === "loading";
  if (loading) return <p>Loading...</p>;

  return (
    <div>
      {session ? (
       <div className='bg-[#e9e9e9] min-h-screen p-8 
       px-[10px] md:px-[160px]'>
         <Form/>
       </div>
      ) : (
        <div>Art Builder</div>
      )}
    </div>
  );
};

export default ArtBuilder;
