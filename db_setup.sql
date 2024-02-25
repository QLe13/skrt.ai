USE skrt;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    credits INT NOT NULL DEFAULT 999
);

CREATE TABLE artworks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    artist_id INT NOT NULL,
    title VARCHAR(255) NOT NULL,
    image_url VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (artist_id) REFERENCES users(id)
);

CREATE TABLE generated_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    prompt TEXT NOT NULL,
    image_url VARCHAR(255) NOT NULL,
    created_by INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id)
);

CREATE TABLE image_credits (
    image_id INT NOT NULL,
    artist_id INT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES generated_images(id),
    FOREIGN KEY (artist_id) REFERENCES users(id),
    PRIMARY KEY (image_id, artist_id)
);
