/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: false,
    images: {
        remotePatterns: [
            { hostname: 'lh3.googleusercontent.com' },
            { 
                protocol: 'https',
                hostname: 'storage.cloud.google.com' 
            },
            { hostname: 'localhost' },
            { hostname: 'storage.googleapis.com' },
        ],
    },
};

export default nextConfig;

