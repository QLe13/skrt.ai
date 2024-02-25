export interface GoogleUserSession {
    user: {
        email: string;
        name: string;
        image: string;
    },
    expires: string;
} 