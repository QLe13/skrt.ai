import db from '../../utils/db';
import storage from '../../utils/GoogleCloudStorage';
import { NextResponse } from 'next/server';
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

export async function POST(request: Request) {
    const formData = await request.formData();
    const file = formData.get('file');
    if (!(file instanceof File)) {
        return new NextResponse(JSON.stringify({ error: 'File not provided or invalid' }), { status: 400 });
    }
    const buffer = await file.arrayBuffer();
    const image = Buffer.from(buffer);
    const bucketName = process.env.BUCKET_NAME;
    const bucket = storage.bucket(bucketName as string);
    const filePath = file.name;
    const now = new Date();
    const timestamp = now.toISOString();
    const uniqueFilePath = `${timestamp}-${filePath}`;
    const encodedFilePath = encodeURIComponent(uniqueFilePath);

    try {
        await bucket.file(uniqueFilePath).save(image)
        const imageUrl = `https://storage.googleapis.com/${bucketName}/${encodedFilePath}`;

        const title = formData.get('title');
        // put image url and title in database
        const userEmail = formData.get('user');
        // Assuming `db.query` returns a promise that resolves with any type
        const result = await db.query('SELECT id FROM users WHERE email = ? LIMIT 1', [userEmail]) as RowDataPacket[];
        const [user] = result[0] as RowDataPacket[]; // Assert that the first element is indeed a RowDataPacket
        

        if (!user) {
            return new NextResponse(JSON.stringify({ error: 'User not found' }), { status: 404 });
        }
        const userId = user.id;
        // put image into artworks table
        await db.query('INSERT INTO artworks (artist_id, title, image_url) VALUES (?, ?, ?)', [userId, title, imageUrl]);
        
        return new NextResponse(JSON.stringify({ message: 'Upload successful' }), { status: 200 });
    } catch (error) {
        return new NextResponse(JSON.stringify({ error: error }), { status: 500 });
    }
}
