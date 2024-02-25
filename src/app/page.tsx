
'use client'
import React, { useState, useEffect } from 'react';
import { GrPaint } from "react-icons/gr";
import { FaUserAstronaut } from "react-icons/fa";
import { IoIosColorPalette } from "react-icons/io";
import { FaCircleChevronDown } from "react-icons/fa6";

const Home: React.FC = () => {
  const [currentSection, setCurrentSection] = useState(0);
  const totalSections = 4; // Update this based on the actual number of sections
  
  // Debounce function to limit the rate at which a function is executed
// Define a generic type for the function to allow any function to be passed with its specific signature
  const debounce = <F extends (...args: any[]) => any>(func: F, delay: number): ((...args: Parameters<F>) => void) => {
    let inDebounce: ReturnType<typeof setTimeout> | null; // Use ReturnType to get the correct type for setTimeout
    return function(this: ThisParameterType<F>, ...args: Parameters<F>) { // Use spread operator for arguments
      const context = this; // Preserve 'this' context
      clearTimeout(inDebounce as NodeJS.Timeout); // Clear previous timeout, casting inDebounce because it can be null
      inDebounce = setTimeout(() => func.apply(context, args), delay);
    };
  };
  

  const handleScroll = (e: WheelEvent) => {
    e.preventDefault();
    if (e.deltaY > 0) {
      // Scroll Down
      setCurrentSection(prev => prev < totalSections - 1 ? prev + 1 : prev);
    } else {
      // Scroll Up
      setCurrentSection(prev => prev > 0 ? prev - 1 : prev);
    }
  };

  // Wrapping handleScroll with debounce
  const debouncedHandleScroll = debounce(handleScroll, 34);

  useEffect(() => {
    const scrollEvent = 'wheel';
    window.addEventListener(scrollEvent, debouncedHandleScroll, { passive: false });

    return () => {
      window.removeEventListener(scrollEvent, debouncedHandleScroll);
    };
  }, [debouncedHandleScroll]); // Make sure to include debouncedHandleScroll here to avoid re-creating the debounced function on every render


  return (
    <div className="relative flex flex-col min-h-full overflow-hidden ">
      {Array.from({ length: totalSections }).map((_, index) => (
        <div
          key={index}
          className={`fixed top-0 left-0 w-full h-full flex flex-col justify-center items-center text-center transition-transform duration-700 ease-in-out ${
            index - currentSection === 0
              ? "translate-y-0"
              : index - currentSection < 0
              ? "-translate-y-full"
              : "translate-y-full"
          }`}
        >
          {index === 0 && (
            <SectionHero />
          )}
          {index === 1 && (
            <SectionFeatures />
          )}
          {index === 2 && (
            <SectionHowItWorks />
          )}
          {index === 3 && (
            <SectionFooter />
          )}
        </div>
      ))}
    </div>
  );
};

const SectionHero = () => (
  <div className="text-white bg-gradient-to-r from-purple-500 to-pink-500 p-12 space-y-4">
    {/* Hero Section Content */}
    <div className="h-screen flex flex-col justify-center items-center text-white text-center bg-gradient-to-r from-purple-500 to-pink-500 p-12 space-y-4">
        <h1 className="text-4xl md:text-5xl font-bold">Welcome to sKrt.ai</h1>
        <p className="text-xl md:text-2xl">Explore the intersection of art and AI technology. Create, innovate, and bring your visions to life with our cutting-edge AI-powered tools. Celebrate creativity and contribute to the arts in a unique and impactful way.</p>
        <button className="mt-4 bg-black text-white hover:bg-gray-700 font-bold py-2 px-4 rounded-full">
          Get Creative
        </button>
    </div>
    <div className="absolute inset-x-0 bottom-0 mb-4 flex justify-center">
      <FaCircleChevronDown className="text-4xl animate-bounce" />
    </div>
  </div>
);

const SectionFeatures = () => (
  <div className="bg-gray-100 flex flex-col justify-center items-center text-center h-full">
    <h2 className="text-3xl font-semibold mb-6">Features</h2>
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 px-4">
      <FeatureCard
        Icon={GrPaint}
        title="AI-Powered Generation"
        description="Generate stunning artworks with the help of AI."
      />
      <FeatureCard
        Icon={FaUserAstronaut}
        title="Artist Appreciation"
        description="Contributing artists receive credits and recognition."
      />
      <FeatureCard
        Icon={IoIosColorPalette}
        title="Creative Freedom"
        description="Let your imagination run wild with limitless possibilities."
      />
    </div>
    <div className="absolute inset-x-0 bottom-0 mb-4 flex justify-center">
      <FaCircleChevronDown className="text-4xl animate-bounce" />
    </div>
  </div>
);

const FeatureCard:React.FC<any> = ({ Icon, title, description }) => (
  <div className="p-6 rounded-lg bg-white shadow-xl">
    <Icon className="mx-auto text-4xl mb-4" />
    <h3 className="text-xl font-semibold mb-2">{title}</h3>
    <p>{description}</p>
  </div>
);

const SectionHowItWorks = () => (
  <div className="bg-gray-100 flex flex-col justify-center items-center text-center h-screen px-4 w-screen">
    <h2 className="text-3xl font-semibold mb-6">How It Works</h2>
    <div className="space-y-4">
      <p className="text-lg">1. Enter your creative prompt</p>
      <p className="text-lg">2. Our AI generates your artwork</p>
      <p className="text-lg">3. Artists behind the AI model get credited</p>
    </div>
    <div className="absolute inset-x-0 bottom-0 mb-4 flex justify-center">
      <FaCircleChevronDown className="text-4xl animate-bounce" />
    </div>
  </div>
);

const SectionFooter = () => (
  <div className="bg-gray-800 text-white flex flex-col justify-center items-center text-center h-screen w-screen text-2xl">
    © 2024 sKrt.ai - All rights reserved.
  </div>
);


export default Home;
