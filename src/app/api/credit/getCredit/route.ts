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


export async function GET(request: Request) {
    const url = new URL(request.url);
    const email = url.searchParams.get('email');
    // get uid of artist
    const result = await db.query('SELECT id FROM users WHERE email = ? LIMIT 1', [email]) as RowDataPacket[];
    const [user] = result[0] as RowDataPacket[]; // Assert that the first element is indeed a RowDataPacket
    if (!user) {
        return new Response('User not found', { status: 404 });
    }
    const artistId = user.id;
    const creditsCount = await db.query('SELECT COUNT(*) FROM image_credits WHERE artist_id = ?', [artistId]) as RowDataPacket[];
    const [credits] = creditsCount[0] as RowDataPacket[];
    return new Response(JSON.stringify({ credits: credits['COUNT(*)'] }), { status: 200 });
}