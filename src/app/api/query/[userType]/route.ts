import { NextResponse } from 'next/server';

// Assuming `db` is your database utility module
import db from '../../../utils/db';
import { RowDataPacket } from 'mysql2';

// USE skrt;

// CREATE TABLE users (
//     id INT AUTO_INCREMENT PRIMARY KEY,
//     email VARCHAR(255) NOT NULL UNIQUE,
//     credits INT NOT NULL DEFAULT 999
// );

// CREATE TABLE artworks (
//     id INT AUTO_INCREMENT PRIMARY KEY,
//     artist_id INT NOT NULL,
//     title VARCHAR(255) NOT NULL,
//     image_url VARCHAR(255) NOT NULL,
//     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
//     FOREIGN KEY (artist_id) REFERENCES users(id)
// );

// CREATE TABLE generated_images (
//     id INT AUTO_INCREMENT PRIMARY KEY,
//     prompt TEXT NOT NULL,
//     image_url VARCHAR(255) NOT NULL,
//     created_by INT NOT NULL,
//     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
//     FOREIGN KEY (created_by) REFERENCES users(id)
// );

// CREATE TABLE image_credits (
//     image_id INT NOT NULL,
//     artist_id INT NOT NULL,
//     FOREIGN KEY (image_id) REFERENCES generated_images(id),
//     FOREIGN KEY (artist_id) REFERENCES users(id),
//     PRIMARY KEY (image_id, artist_id)
// );


interface UserType {
    userType: 'artist' | 'user';
}

export async function GET(request: Request, { params }:{ params: { userType: string } }) {
    const url = new URL(request.url, 'http://localhost');
    const email = url.searchParams.get('email');
    const userType = params.userType;
    
    // console.log(userType, email);
    // // Proceed with your database query logic here.
    
    // return new NextResponse(JSON.stringify({ userType, email }), {
    //     status: 200,
    // });
    if(userType === 'artist'){
        // get id of artist
        const result = await db.query('SELECT id FROM users WHERE email = ? LIMIT 1', [email]) as RowDataPacket[];
        const [user] = result[0] as RowDataPacket[]; // Assert that the first element is indeed a RowDataPacket
        if (!user) {
            return new NextResponse(JSON.stringify({ error: 'User not found' }), { status: 404 });
        }
        const artistId = user.id;
        // get all artworks of artist
        const artworksPacket = await db.query('SELECT * FROM artworks WHERE artist_id = ?', [artistId]) as RowDataPacket[];
        const artworks = artworksPacket[0] as RowDataPacket[];
        return new NextResponse(JSON.stringify(artworks), { status: 200 });
    }
    if(userType === 'user'){
        // get id of user
        const result = await db.query('SELECT id FROM users WHERE email = ? LIMIT 1', [email]) as RowDataPacket[];
        const [user] = result[0] as RowDataPacket[]; // Assert that the first element is indeed a RowDataPacket
        if (!user) {
            return new NextResponse(JSON.stringify({ error: 'User not found' }), { status: 404 });
        }
        const userId = user.id;
        // get all generated images of user
        const imagesPacket = await db.query('SELECT * FROM generated_images WHERE created_by = ?', [userId]) as RowDataPacket[];
        const images = imagesPacket[0] as RowDataPacket[];
        return new NextResponse(JSON.stringify(images), { status: 200 });
    }
}
